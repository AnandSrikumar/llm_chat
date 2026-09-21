from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from app.chunkers.chunk_base import AsyncChunker, SyncChunker
from app.chunkers.docx_chunker import DocxChunker
from app.chunkers.pdf_chunker import PdfChunker
from app.chunkers.text_splitter import TextChunker
from app.core.config import Settings

CHUNK_MAPPER = {
    ".pdf": PdfChunker,
    ".txt": TextChunker,
    ".md": TextChunker,
    ".docx": DocxChunker,
}


async def chunk_file(extension: str, file_data: bytes, settings: Settings):
    chunk_type = CHUNK_MAPPER.get(extension)
    if not chunk_type:
        raise HTTPException(status_code=400, detail="unsupported file type")
    chunk_obj: SyncChunker | AsyncChunker = chunk_type(settings)
    if isinstance(chunk_obj, SyncChunker):
        return await run_in_threadpool(
            chunk_obj.chunk,
            file_data,
        )
    return await chunk_obj.chunk(file_data)
