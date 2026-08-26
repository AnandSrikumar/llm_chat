from io import BytesIO
import uuid

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
import pymupdf4llm
import pymupdf
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

import sentence_transformers

from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.service.text_services import clean_chunks_for_bm25
from app.core.exceptions import UnsupportedFormatError

logger = get_logger(__name__)

MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


def generate_file_name(content_type: str) -> str:
    extension = MIME_EXTENSIONS.get(content_type)

    if extension is None:
        logger.warning("Unsupported file content type received (mime_type=%s)", content_type)
        raise UnsupportedFormatError()

    return f"{uuid.uuid4()}{extension}"


def create_pdf_chunks(
    file: UploadFile,
    markdown_splitter: MarkdownHeaderTextSplitter,
    recursive_splitter: RecursiveCharacterTextSplitter,
):
    if file.content_type != "application/pdf":
        logger.warning(
            "PDF chunking rejected for unsupported content type (mime_type=%s)",
            file.content_type,
        )
        raise UnsupportedFormatError()

    logger.info("Starting PDF chunking (size=%s)", file.size)
    pdf_bytes = file.file.read()
    pdf_bytes.seek(0)
    doc = pymupdf.open(
        stream=pdf_bytes.getvalue(),
        filetype="pdf",
    )
    try:
        markdown = pymupdf4llm.to_markdown(doc)
        sections = markdown_splitter.split_text(markdown)
        chunks = recursive_splitter.split_text(sections)
        logger.info(
            "PDF chunking completed (sections=%s, chunks=%s)",
            len(sections),
            len(chunks),
        )
    except Exception:
        logger.exception("PDF chunking failed")
        raise
    finally:
        doc.close()
    return chunks


def process_pdf(
    file: UploadFile,
    markdown_splitter: MarkdownHeaderTextSplitter,
    recursive_splitter: RecursiveCharacterTextSplitter,
):
    logger.info("Preparing PDF chunks for BM25 indexing")
    chunks = create_pdf_chunks(
        file,
        markdown_splitter,
        recursive_splitter,
    )
    clean_chunks = clean_chunks_for_bm25(chunks)
    logger.info("PDF chunks prepared for BM25 indexing (chunks=%s)", len(chunks))
    return chunks, clean_chunks


async def persist_file_and_chunks(
    db,
    *,
    conversation_id: int,
    mime_type: str,
    size: int,
    file_name_original: str,
    file_name: str,
    chunks: list,
    clean_chunks: list,
):
    logger.info(
        "Persisting uploaded file and chunks (conversation_id=%s, mime_type=%s, size=%s, chunks=%s)",
        conversation_id,
        mime_type,
        size,
        len(chunks),
    )
    async with db.transaction():
        file_row = await db.fetchrow(
            """
            INSERT INTO files (
                conversation_id,
                mime_type,
                size,
                file_name_original,
                file_name
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            conversation_id,
            mime_type,
            size,
            file_name_original,
            file_name,
        )

        file_id = file_row["id"]

        chunk_records = [
            (
                file_id,
                index,
                None,
                chunk,
                clean_chunks[index],
            )
            for index, chunk in enumerate(chunks)
        ]

        await db.executemany(
            """
            INSERT INTO chunks (
                file_id,
                chunk_index,
                page_number,
                chunk_text,
                bm25_chunk_text
            )
            VALUES ($1, $2, $3, $4, $5)
            """,
            chunk_records,
        )

    logger.info(
        "Uploaded file and chunks persisted (conversation_id=%s, file_id=%s, chunks=%s)",
        conversation_id,
        file_id,
        len(chunk_records),
    )
    return file_id


async def ingest_pdf(
    file: UploadFile,
    markdown_splitter: MarkdownHeaderTextSplitter,
    recursive_splitter: RecursiveCharacterTextSplitter,
    conversation_id: int,
    db: PgClient,
):
    logger.info(
        "Starting PDF ingestion (conversation_id=%s, mime_type=%s, size=%s)",
        conversation_id,
        file.content_type,
        file.size,
    )
    chunks, clean_chunks = await run_in_threadpool(
        process_pdf,
        file,
        markdown_splitter,
        recursive_splitter,
    )
    file_id = await persist_file_and_chunks(
        db,
        conversation_id=conversation_id,
        mime_type=file.content_type,
        size=file.size,
        file_name_original=file.filename,
        file_name=generate_file_name(file.content_type),
        chunks=chunks,
        clean_chunks=clean_chunks,
    )
    logger.info(
        "PDF ingestion completed (conversation_id=%s, file_id=%s, chunks=%s)",
        conversation_id,
        file_id,
        len(chunks),
    )
    return file_id
