import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.core.prompts import (
    COMPACTION_PROMPT,
    IMAGE_DESCRIBE,
    NAME_GENERATOR_PROMPT,
    SYSTEM_PROMPT,
)
from app.llm.llm_base import LLMBase
from app.service.db_queries import (
    INSERT_CONVERSATION,
    SIMILAR_CHUNKS,
    UPDATE_CONVERSATION,
)
from app.tokenizers.encoders import Encoder

logger = get_logger(__name__)

CONVERSATION_LOCKS: dict[int, asyncio.Lock] = {}


@dataclass
class ChatMeta:
    convo_name: str
    messages: list[str]
    compaction: list[str]


async def get_chat_meta(pg: PgClient, conversation_id: int | None):
    if not conversation_id:
        logger.info("Initializing metadata for a new conversation")
        return ChatMeta(convo_name="", messages=[], compaction=[])
    logger.info("Loading conversation metadata (conversation_id=%s)", conversation_id)
    query = "select convo_name, messages, compaction from conversations where id=$1"
    result = await pg.fetchone(query, conversation_id)
    if result is None:
        logger.warning(
            "Conversation was not found (conversation_id=%s)", conversation_id
        )
        raise ValueError("Conversation not found")
    logger.info("Conversation metadata loaded (conversation_id=%s)", conversation_id)
    result = dict(result)
    result["messages"] = json.loads(result.get("messages", "[]"))
    result["compaction"] = json.loads(result.get("compaction", "[]"))
    return ChatMeta(**result)


async def create_chat_name(llm: LLMBase, model_name: str, message: str):
    logger.info(
        "Generating conversation name (model=%s, message_length=%s)",
        model_name,
        len(message),
    )
    try:
        res = await llm.generate(
            max_tokens=500,
            model=model_name,
            input=message,
            instructions=NAME_GENERATOR_PROMPT,
        )
    except Exception as e:
        logger.exception(
            "Conversation name generation failed (model=%s), error: %s",
            model_name,
            str(e),
        )
        raise
    logger.info("Conversation name generated")
    return res.output_text


async def create_conversation(user_id: int, chat_name: str, pg: PgClient):
    logger.info("Persisting new conversation (user_id=%s)", user_id)
    res = await pg.fetchone(
        INSERT_CONVERSATION,
        user_id,
        chat_name,
    )
    conversation_id = dict(res)["id"]
    logger.info(
        "New conversation persisted (conversation_id=%s, user_id=%s)",
        conversation_id,
        user_id,
    )
    return conversation_id


async def count_tokens(
    model_name: str,
    messages: list,
    encoding: Encoder,
) -> int:
    logger.debug(
        "Counting context input tokens (model=%s, item_count=%s)",
        model_name,
        len(messages),
    )
    tot = 0
    for message in messages:
        if isinstance(message, dict):
            if "content" not in message:
                tot += len(encoding.encode(json.dumps(message)))
                continue
            tot += len(encoding.encode(message["content"]))
            continue
        tot += len(encoding.encode(message))
    return tot


async def compact_messages(llm: LLMBase, model_name: str, messages: list) -> dict:
    old_messages = messages[:-6]
    recent_messages = messages[-6:]
    logger.info(
        "Compacting conversation context (messages_to_summarize=%s, recent_messages=%s, model=%s)",
        len(old_messages),
        len(recent_messages),
        model_name,
    )
    summary_input = [
        {
            "role": "user",
            "content": (f"{COMPACTION_PROMPT}\n\n" f"Conversation:\n{old_messages}"),
        }
    ]

    try:
        response = await llm.generate(
            model=model_name,
            input=summary_input,
            max_output_tokens=4000,
        )
    except Exception:
        logger.exception(
            "Conversation context compaction failed (model=%s)", model_name
        )
        raise

    summary = response.output_text
    logger.info("Conversation context compaction completed")

    return [
        {
            "role": "system",
            "content": f"Conversation summary:\n{summary}",
        },
        *recent_messages,
    ]


async def search_rag(
    query: str,
    chat_id: int,
    pg: PgClient,
    embed_model: Encoder,
    top_k: int = 3,
    threshold: float = 0.4,
):
    embeds = embed_model.encode(query)
    similar = await pg.fetch(SIMILAR_CHUNKS, embeds, chat_id, top_k)
    context = []
    for c in similar:
        chunk = f"file name: {c['filename_original']}\nscore: {c['cosine_distance']}\ntext: {c['chunk_text']}"
        context.append(chunk)
        logger.info(
            f"filename: {c['filename_original']}-->chunk score: {c['cosine_distance']}"
        )
    return "\n\n".join(context)


async def generate_message(
    llm: LLMBase,
    model_name: str,
    pg: PgClient,
    conversation_id: int,
    chat_meta: ChatMeta,
    max_tokens: int = 1024,
    lock: asyncio.Lock = None,
) -> AsyncGenerator[str, None]:
    logger.info(
        "Requesting streamed LLM response (conversation_id=%s, model=%s, max_tokens=%s)",
        conversation_id,
        model_name,
        max_tokens,
    )
    try:
        stream = await llm.stream(
            model=model_name,
            input=chat_meta.compaction,
            instructions=SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        assistant_chunks: list[str] = []
        yield f"chat_id: {conversation_id}\n\n"
        async for event in stream:
            # logger.info(f"{event.type}: {event}")
            if event.type == "response.output_text.delta":
                assistant_chunks.append(event.delta)
                yield f"{event.delta}"
            elif event.type == "response.completed":
                logger.info(
                    "status=%s incomplete_details=%r usage=%r",
                    event.response.status,
                    event.response.incomplete_details,
                    event.response.usage,
                )
                break

        assistant_message = "".join(assistant_chunks)
        logger.info(
            "LLM response stream completed (conversation_id=%s, response_length=%s)",
            conversation_id,
            len(assistant_message),
        )
        chat_meta.messages.append({"role": "assistant", "content": assistant_message})
        chat_meta.compaction.append({"role": "assistant", "content": assistant_message})

        async with pg.transaction() as conn:
            await conn.execute(
                UPDATE_CONVERSATION,
                json.dumps(chat_meta.messages),
                json.dumps(chat_meta.compaction),
                conversation_id,
            )
        logger.info(
            "Conversation response persisted (conversation_id=%s)", conversation_id
        )
    except Exception as e:
        logger.exception(
            "Chat response generation failed (conversation_id=%s) %s",
            conversation_id,
            e,
        )
        raise
    finally:
        if lock is not None:
            lock.release()


def get_conversation_lock(conversation_id: int) -> asyncio.Lock:
    lock = CONVERSATION_LOCKS.get(conversation_id)

    if lock is None:
        lock = asyncio.Lock()
        CONVERSATION_LOCKS[conversation_id] = lock

    return lock
