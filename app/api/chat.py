import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer

from app.core.deps import (
    COMPACT_THRESHOLD,
    EMBEDDING_MODEL,
    LLM,
    LLM_MODEL,
    SPLITTERS,
    STORAGE_TYPE,
    TIKTOKEN_ENCODING,
    USER,
    Pg,
)
from app.core.exceptions import LLMGenerationError
from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.core.splitters import Splitters
from app.service.chat_service import (
    compact_messages,
    count_tokens,
    create_chat_name,
    create_conversation,
    generate_message,
    get_chat_meta,
    get_conversation_lock,
)
from app.service.file_services import persist_text_file
from app.storage.storage_base import Storage

router = APIRouter()
logger = get_logger(__name__)


async def _handle_files(
    files: list[UploadFile],
    splitters: Splitters,
    embedding_model: SentenceTransformer,
    chat_id: int,
    storage_type: Storage,
    pg: PgClient,
):
    if not files:
        return []
    tasks = [
        persist_text_file(file, splitters, embedding_model, chat_id, storage_type, pg)
        for file in files
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.error("Failed to persist file: %s", result)
    logger.info(f"gathering file texts...{len(results)}")
    return [r for r in results if not isinstance(r, Exception)]


@router.post("/v1/chat")
async def chat(
    message: Annotated[str, Form()],
    llm: LLM,
    llm_model: LLM_MODEL,
    pg: Pg,
    user: USER,
    compact_threshold: COMPACT_THRESHOLD,
    tiktoken_encoding: TIKTOKEN_ENCODING,
    splitters: SPLITTERS,
    storage_type: STORAGE_TYPE,
    embedding_model: EMBEDDING_MODEL,
    files: Annotated[list[UploadFile] | None, File()] = None,
    max_tokens: int | None = 1024,
    chat_id: int | None = None,
):
    if chat_id is None:
        logger.info("Creating a new conversation for user_id=%s", user["id"])
        name = await create_chat_name(
            llm, llm_model, [{"role": "user", "content": message}]
        )
        chat_id = await create_conversation(user["id"], name, pg)
        logger.info(
            "Created conversation_id=%s, convo_name=%s for user_id=%s",
            chat_id,
            name,
            user["id"],
        )
    lock = get_conversation_lock(chat_id)
    if lock.locked():
        raise LLMGenerationError()
    await lock.acquire()
    try:
        user_messages = []
        file_texts = await _handle_files(
            files, splitters, embedding_model, chat_id, storage_type, pg
        )
        for text in file_texts:
            user_messages.append({"role": "user", "content": text})
        logger.info(
            "Chat request received (user_id=%s, conversation_id=%s, message_length=%s, has_file=%s, max_tokens=%s)",
            user["id"],
            chat_id,
            len(message),
            files is not None,
            max_tokens,
        )
        user_messages.append({"role": "user", "content": message})
        chat_meta = await get_chat_meta(pg, chat_id)
        chat_meta.messages.extend(user_messages)
        chat_meta.compaction.extend(user_messages)

        num_tokens = await count_tokens(
            llm_model, chat_meta.compaction, tiktoken_encoding
        )
        logger.info(
            "Conversation context evaluated (conversation_id=%s, input_tokens=%s, compact_threshold=%s)",
            chat_id,
            num_tokens,
            compact_threshold,
        )
        if num_tokens > compact_threshold:
            logger.info("Compacting conversation context (conversation_id=%s)", chat_id)
            chat_meta.compaction = await compact_messages(
                llm, llm_model, chat_meta.compaction
            )
        logger.info("Starting response stream (conversation_id=%s)", chat_id)
    except Exception as e:
        lock.release()
        logger.error(f"{e} has occured....")
        raise
    return StreamingResponse(
        generate_message(llm, llm_model, pg, chat_id, chat_meta, max_tokens, lock),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
