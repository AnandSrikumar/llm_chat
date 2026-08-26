from io import BytesIO

from fastapi import UploadFile
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

logger = get_logger(__name__)


def create_pdf_chunks(
    file: UploadFile,
    markdown_splitter: MarkdownHeaderTextSplitter,
    recursive_splitter: RecursiveCharacterTextSplitter,
):
    if file.content_type != "application/pdf":
        ...
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
    except Exception as e:
        logger.info(f"Failed to chunk the pdf: {e}")
    finally:
        doc.close()
    return chunks


async def ingest_pdf(
    file: UploadFile,
    markdown_splitter: MarkdownHeaderTextSplitter,
    recursive_splitter: RecursiveCharacterTextSplitter,
    db: PgClient,
):
    chunks = create_pdf_chunks(
        file,
        markdown_splitter,
        recursive_splitter,
    )
    chunks = clean_chunks_for_bm25(chunks)
    