import hashlib
from io import BytesIO
from typing import Any
import uuid

import asyncpg
from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
import pymupdf4llm
import pymupdf
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from sentence_transformers import SentenceTransformer

from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.service.db_queries import CHUNK_INSERT_QUERY, FILE_INSERT_QUERY, FILE_OWNER_QUERY
from app.service.text_services import clean_chunks_for_bm25
from app.core.exceptions import NotFound, UnsupportedFormatError
from app.storage.storage_base import Storage

logger = get_logger(__name__)

MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


class FileObject:
    filename: str
    chunks: list[str]
    cleaned_chunks: list[str]
    embeds: list[float]
    file_original_name: str
    mime_type: str
    size: str
    content_hash: str
    data: str
    owner_id: int


def generate_file_name(content_type: str) -> str:
    extension = MIME_EXTENSIONS.get(content_type)

    if extension is None:
        logger.warning(
            "Unsupported file content type received (mime_type=%s)", content_type
        )
        raise UnsupportedFormatError()

    return f"{uuid.uuid4()}{extension}"


def _chunk_pdf(
    data: bytes,
    md_splitter: MarkdownHeaderTextSplitter,
    rec_splitter: RecursiveCharacterTextSplitter,
):
    data.seek(0)
    doc = pymupdf.open(
        stream=data.getvalue(),
        filetype="pdf",
    )
    try:
        markdown = pymupdf4llm.to_markdown(doc)
        sections = md_splitter.split_text(markdown)
        chunks = rec_splitter.split_text(sections)
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


def _chunk_docx(file_data: BytesIO, rec_splitter: RecursiveCharacterTextSplitter): ...


def _chunk_txt(file_data: bytes, rec_splitter: RecursiveCharacterTextSplitter):
    file_data.seek(0)
    chunks = rec_splitter.split_text(file_data)
    return chunks


def chunk_file(
    data: bytes,
    mime_type: str,
    md_splitter: MarkdownHeaderTextSplitter,
    rec_splitter: RecursiveCharacterTextSplitter,
):
    match mime_type:
        case "application/pdf":
            return _chunk_pdf(data, md_splitter, rec_splitter)
        case "text/plain":
            return _chunk_txt(data, rec_splitter)
        case "text/markdown":
            return _chunk_txt(data, rec_splitter)
        case _:
            raise UnsupportedFormatError()


def process_text_file(
    file: UploadFile,
    md_splitter: MarkdownHeaderTextSplitter,
    rec_splitter: RecursiveCharacterTextSplitter,
    embed_model: SentenceTransformer,
) -> FileObject:
    mime_type = file.content_type
    file_name = generate_file_name(mime_type)
    data = file.file.read()
    size = file.size
    content_hash = hashlib.sha256(data).hexdigest()
    chunks = chunk_file(data, mime_type, md_splitter, rec_splitter)
    cleaned_chunks = clean_chunks_for_bm25(chunks)
    embeds = embed_model.encode(
        chunks,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return FileObject(
        filename=file_name,
        chunks=chunks,
        cleaned_chunks=cleaned_chunks,
        embeds=embeds,
        file_original_name=file.filename,
        mime_type=mime_type,
        size=size,
        content_hash=content_hash,
        data=data
    )


async def _insert_file(file_obj: FileObject,
                       storage: Storage,
                       conversation_id: int,
                       conn: asyncpg.Connection):
    dir_res = await conn.fetchrow(FILE_OWNER_QUERY,
                                  conversation_id)
    if not dir_res:
        raise NotFound("User or conversation not found")
    path = f"{dir_res['username']}/{dir_res['convo_dir']}/{dir_res['file_gen_name']}"
    res = await conn.fetchrow(
        FILE_INSERT_QUERY,
        file_obj.file_original_name,
        file_obj.filename,
        file_obj.content_hash,
        file_obj.mime_type,
        file_obj.size,
        path,
        storage.storage_type
    )
    return res['id'], dir_res['username']


async def _insert_chunk(file_obj: FileObject, file_id, conn: asyncpg.Connection):
    chunks = file_obj.chunks
    cleaned_chunks = file_obj.cleaned_chunks
    embeds = file_obj.embeds
    records = [
        (file_id, chunk, cleaned_chunk, embedding)
        for chunk, cleaned_chunk, embedding
        in zip(chunks, cleaned_chunks, embeds)
    ]
    await conn.executemany(CHUNK_INSERT_QUERY, records)
    

async def persist_text_file(
    file: UploadFile,
    md_splitter: MarkdownHeaderTextSplitter,
    rec_splitter: RecursiveCharacterTextSplitter,
    embed_model: SentenceTransformer,
    conversation_id: int,
    storage: Storage,
    pg: PgClient
): 
    file_obj = await run_in_threadpool(
        process_text_file,
        file,
        md_splitter,
        rec_splitter,
        embed_model
    )
    async with pg.transaction() as conn:
        file_id, owner_id = await _insert_file(
            file_obj,
            storage,
            conversation_id,
            conn
        )
        await _insert_chunk(file_obj, file_id, conn)
        storage.save_file(
            file_obj.data,
            file_obj.filename,
            owner_id,
            conversation_id
        )

