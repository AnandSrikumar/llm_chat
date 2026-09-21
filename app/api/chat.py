import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.core.deps import LLM, SETTINGS, SPLITTERS, STORAGE_TYPE, USER, Pg
from app.core.exceptions import LLMGenerationError
from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.core.prompts import FILE_DESCRIPTION, IMAGE_DESCRIBE
from app.core.splitters import Splitters
from app.llm.llm_base import LLMBase
from app.llm.llm_initiate import LLMModelObject
from app.llm.openai_llm import OpenAILLM
from app.service.chat_service import (
    ChatMeta,
    compact_messages,
    count_tokens,
    create_chat_name,
    create_conversation,
    generate_message,
    get_chat_meta,
    get_conversation_lock,
    search_rag,
)
from app.service.file_operations import file_pipeline
from app.storage.storage_base import Storage
from app.tokenizers.encoders import Encoder

router = APIRouter()
logger = get_logger(__name__)


async def _create_conversation(
    user: dict, pg: PgClient, model_obj: LLMBase, llm_model: str, message: str
):
    logger.info("Creating a new conversation for user_id=%s", user["id"])
    name = await create_chat_name(
        model_obj, llm_model, [{"role": "user", "content": message}]
    )
    chat_id = await create_conversation(user["id"], name, pg)
    logger.info(
        "Created conversation_id=%s, convo_name=%s for user_id=%s",
        chat_id,
        name,
        user["id"],
    )
    return chat_id


async def _prepare_chat_history(pg: PgClient, chat_id: int, message: str) -> ChatMeta:
    persisted_chat = await get_chat_meta(pg, chat_id)
    persisted_chat.compaction.append({"role": "user", "content": message})
    persisted_chat.messages.append({"role": "user", "content": message})
    return persisted_chat


def _count_tokens(encoder: Encoder, compaction: list[dict]):
    logger.info(f"counting tokens of: {compaction}")
    texts = [msg["content"] for msg in compaction]
    return encoder.count_tokens(texts)


async def _handle_files(
    files: list[UploadFile] | None,
    chat_id: int,
    settings: Settings,
    storage: Storage,
    llm_model_obj: LLMModelObject,
    pg: PgClient,
) -> str | None:
    if not files:
        return ""

    logger.info(
        "Processing %s uploaded files for conversation_id=%s",
        len(files),
        chat_id,
    )

    results = await asyncio.gather(
        *(
            file_pipeline(
                chat_id=chat_id,
                file=file,
                settings=settings,
                storage=storage,
                llm=llm_model_obj,
                pg=pg,
            )
            for file in files
        )
    )

    file_contents = "\n\n".join(
        f"Filename: {filename}\n" f"Content:\n{content}"
        for filename, content in results
    )

    return FILE_DESCRIPTION.format(
        files=file_contents,
    )


@router.post("/v1/chat")
async def chat(
    message: Annotated[str, Form()],
    llm: LLM,
    pg: Pg,
    user: USER,
    storage_type: STORAGE_TYPE,
    settings: SETTINGS,
    files: Annotated[list[UploadFile] | None, File()] = None,
    chat_id: int | None = None,
    llm_model: str = "ministral-3:3b",
):
    if llm_model not in llm:
        raise HTTPException(status_code=500, detail="Failed to load LLM model")

    model_obj = llm[llm_model].llm_object
    encoder_obj = llm[llm_model].encoding_object

    if chat_id is None:
        chat_id = await _create_conversation(user, pg, model_obj, llm_model, message)

    lock = get_conversation_lock(chat_id)
    if lock.locked():
        raise LLMGenerationError()
    await lock.acquire()

    try:
        file_context = await _handle_files(
            files, chat_id, settings, storage_type, llm[llm_model], pg
        )

        rag_context = await search_rag(message, chat_id, pg, encoder_obj)
        persisted_chat = await _prepare_chat_history(pg, chat_id, message)

        persisted_chat.compaction.append({"role": "user", "content": rag_context})
        persisted_chat.messages.append({"role": "user", "content": rag_context})


        persisted_chat.compaction.append({"role": "user", "content": file_context})
        persisted_chat.messages.append({"role": "user", "content": file_context})
        if (
            _count_tokens(encoder_obj, persisted_chat.compaction)
            > settings.compact_threshold
        ):
            logger.info(f"conversation is compacting...")
            persisted_chat.compaction = await compact_messages(
                model_obj, llm_model, persisted_chat.compaction
            )

    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        lock.release()
        raise

    return StreamingResponse(
        generate_message(
            model_obj, llm_model, pg, chat_id, persisted_chat, settings.max_tokens, lock
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
