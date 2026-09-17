import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.core.deps import (
    LLM,
    SETTINGS,
    SPLITTERS,
    STORAGE_TYPE,
    USER,
    Pg,
)
from app.core.exceptions import LLMGenerationError
from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.core.prompts import FILE_DESCRIPTION, IMAGE_DESCRIBE
from app.core.splitters import Splitters
from app.llm.llm_base import LLMBase
from app.service.chat_service import (
    compact_messages,
    count_tokens,
    create_chat_name,
    create_conversation,
    generate_message,
    get_chat_meta,
    get_conversation_lock,
    search_rag,
)

from app.llm.openai_llm import OpenAILLM

from app.storage.storage_base import Storage

router = APIRouter()
logger = get_logger(__name__)


async def _create_conversation(user: dict, 
                               pg: PgClient, 
                               model_obj: LLMBase, 
                               llm_model: str, 
                               message: str):
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


@router.post("/v1/chat")
async def chat(
    message: Annotated[str, Form()],
    llm: LLM,
    pg: Pg,
    user: USER,
    splitters: SPLITTERS,
    storage_type: STORAGE_TYPE,
    settings: SETTINGS,
    files: Annotated[list[UploadFile] | None, File()] = None,
    chat_id: int | None = None,
    llm_model: str = "ministral-3:3b"
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
        user_messages = []
        rag_context = await search_rag(message, chat_id, pg, encoder_obj)

    except Exception as e:
        logger.error(f"LLM chat failed: {e}")
        lock.release()
        raise

    
    return {"status": "ok"}
