from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.core.deps import COMPACT_THRESHOLD, LLM, LLM_MODEL, TIKTOKEN_ENCODING, USER, Pg
from app.core.log import get_logger
from app.service.chat_service import (
    compact_messages,
    count_tokens,
    create_chat_name,
    create_conversation,
    generate_message,
    get_chat_meta,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post("/v1/chat")
async def chat(
    message: Annotated[str, Form()],
    llm: LLM,
    llm_model: LLM_MODEL,
    pg: Pg,
    user: USER,
    compact_threshold: COMPACT_THRESHOLD,
    tiktoken_encoding: TIKTOKEN_ENCODING,
    file: Annotated[UploadFile | None, File()] = None,
    max_tokens: int | None = 1024,
    conversation_id: int | None = None,
):
    logger.info(
        "Chat request received (user_id=%s, conversation_id=%s, message_length=%s, has_file=%s, max_tokens=%s)",
        user["id"],
        conversation_id,
        len(message),
        file is not None,
        max_tokens,
    )
    message = {"role": "user", "content": message}
    chat_meta = await get_chat_meta(pg, conversation_id)
    chat_meta.messages.append(message)
    chat_meta.compaction.append(message)
    chat_id = conversation_id
    if len(chat_meta.messages) == 1:
        logger.info("Creating a new conversation for user_id=%s", user["id"])
        name = await create_chat_name(llm, llm_model, [message])
        chat_meta.convo_name = name
        chat_id = await create_conversation(user["id"], chat_meta, pg)
        logger.info("Created conversation_id=%s for user_id=%s", chat_id, user["id"])
    num_tokens = await count_tokens(llm_model, chat_meta.compaction, tiktoken_encoding)
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
    return StreamingResponse(
        generate_message(llm, llm_model, pg, chat_id, chat_meta, max_tokens),
        media_type="text/event-stream",
    )
