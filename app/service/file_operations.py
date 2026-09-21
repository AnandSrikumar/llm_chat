import hashlib
import uuid
from dataclasses import dataclass

import asyncpg
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.chunkers.chunk_factory import chunk_file
from app.core.config import Settings
from app.core.exceptions import NotFound
from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.llm.llm_initiate import LLMModelObject
from app.service.db_queries import (
    CHUNK_INSERT_QUERY,
    FILE_INSERT_QUERY,
    FILE_OWNER_QUERY,
    FILE_STORAGE_ID_QUERY,
)
from app.service.text_services import clean_chunks_for_bm25
from app.storage.storage_base import Storage

logger = get_logger(__name__)


MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


@dataclass
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


async def _process_file(
    file: UploadFile, settings: Settings, llm: LLMModelObject
) -> FileObject:
    extension = MIME_EXTENSIONS.get(file.content_type)
    if not extension:
        raise HTTPException(
            status_code=400, detail=f"{file.filename} format not supported"
        )
    file_name = f"{uuid.uuid4()}{extension}"
    size = file.size
    data = await file.read()
    content_hash = hashlib.sha256(data).hexdigest()

    chunks = await chunk_file(extension, data, settings)
    logger.info(f"Chunks created for {file.filename}")
    cleaned_chunks = clean_chunks_for_bm25(chunks)
    logger.info(f"cleaned chunks created for {file.filename}")
    texts = [chunk.page_content for chunk in chunks]

    embeds = await run_in_threadpool(
        llm.encoding_object.encode,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.info(f"embeds created for {file.filename}")
    return FileObject(
        filename=file_name,
        chunks=chunks,
        cleaned_chunks=cleaned_chunks,
        embeds=embeds,
        file_original_name=file.filename,
        mime_type=file.content_type,
        size=size,
        content_hash=content_hash,
        data=data,
    )


async def _insert_file(
    file_obj: FileObject, chat_id: int, storage: Storage, conn: asyncpg.Connection
):
    dir_res = await conn.fetchrow(FILE_OWNER_QUERY, chat_id)
    if not dir_res:
        raise NotFound("User or conversation not found")
    path = f"{dir_res['username']}/{dir_res['convo_dir']}/{file_obj.filename}"
    file_storage_id = await conn.fetchrow(FILE_STORAGE_ID_QUERY, storage.storage_type)
    if not file_storage_id:
        raise NotFound("Invalid found storage type")
    res = await conn.fetchrow(
        FILE_INSERT_QUERY,
        chat_id,
        file_obj.file_original_name,
        file_obj.filename,
        file_obj.content_hash,
        file_obj.mime_type,
        file_obj.size,
        path,
        file_storage_id["id"],
    )
    return {"file_id": res["id"], "owner": dir_res["username"]}


async def _insert_chunk(
    file_obj: FileObject,
    file_id,
    conn: asyncpg.Connection,
    llm: LLMModelObject,
):
    chunks = file_obj.chunks
    cleaned_chunks = file_obj.cleaned_chunks
    embeds = file_obj.embeds

    records = [
        (
            file_id,
            idx,
            chunk.page_content,
            cleaned_chunk.page_content,
            embedding,
            llm.encoding_model_name                        
        )
        for idx, (chunk, cleaned_chunk, embedding) in enumerate(
            zip(chunks, cleaned_chunks, embeds)
        )
    ]

    await conn.executemany(CHUNK_INSERT_QUERY, records)


async def file_pipeline(
    chat_id: int,
    file: UploadFile,
    settings: Settings,
    storage: Storage,
    llm: LLMModelObject,    
    pg: PgClient,
):
    try:
        logger.info(f"Persisting file for {chat_id} filename:{file.filename}")
        file_object = await _process_file(file, settings, llm)
        logger.info(f"file object created for {chat_id} filename:{file.filename}")
        async with pg.transaction() as conn:
            file_insert_meta = await _insert_file(file_object, chat_id, storage, conn)
            _ = await _insert_chunk(file_object, file_insert_meta["file_id"], conn, llm)
            storage_key = await storage.save_file(
                file_object.data,
                file_object.filename,
                file_insert_meta["owner"],
                chat_id,
            )
            logger.info(
                "File content persisted (conversation_id=%s, file_id=%s, storage_type=%s, storage_key=%s)",
                chat_id,
                file_insert_meta["file_id"],
                storage.storage_type,
                storage_key,
            )
            return file_object.file_original_name, "\n".join(
                [chunk.page_content for chunk in file_object.chunks]
            )
    except HTTPException as e:
        logger.exception(
            "File processing failed (conversation_id=%s, filename=%s)",
            chat_id,
            file.filename,
        )
        raise
    except NotFound as e:
        logger.exception(
            "File processing failed due to chat not found (conversation_id=%s, filename=%s)",
            chat_id,
            file.filename,
        )
        raise
    except Exception as e:
        logger.exception(
            "File processing failed (conversation_id=%s, filename=%s) error=%s",
            chat_id,
            file.filename,
            e,
        )
        raise HTTPException(status_code=500, detail="Unknown server error")
