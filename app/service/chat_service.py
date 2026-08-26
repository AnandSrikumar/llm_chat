import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from openai import AsyncOpenAI
from transformers import PreTrainedTokenizerBase

from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.core.prompts import (COMPACTION_PROMPT, NAME_GENERATOR_PROMPT,
                              SYSTEM_PROMPT)

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
    result["messages"] = json.loads(result.get("messages", '[]'))
    result["compaction"] = json.loads(result.get("compaction", '[]'))
    return ChatMeta(**result)


async def create_chat_name(llm: AsyncOpenAI, model_name: str, message: str):
    logger.info(
        "Generating conversation name (model=%s, message_length=%s)",
        model_name,
        len(message),
    )
    try:
        res = await llm.responses.create(
            max_output_tokens=500,
            model=model_name,
            input=message,
            instructions=NAME_GENERATOR_PROMPT,
        )
    except Exception:
        logger.exception("Conversation name generation failed (model=%s)", model_name)
        raise
    logger.info("Conversation name generated")
    return res.output_text


async def create_conversation(user_id: int, chat_name: str, pg: PgClient):
    logger.info("Persisting new conversation (user_id=%s)", user_id)
    query = """
        insert into conversations
        (owner_id, convo_name) values 
        ($1, $2) returning id
    """
    res = await pg.fetchone(
        query,
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
    encoding: PreTrainedTokenizerBase,
) -> int:
    logger.debug(
        "Counting context input tokens (model=%s, item_count=%s)",
        model_name,
        len(messages),
    )
    total_tokens = sum(len(encoding.encode(message["content"])) for message in messages)
    return total_tokens


async def compact_messages(llm: AsyncOpenAI, model_name: str, messages: list) -> dict:
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
        response = await llm.responses.create(
            model=model_name,
            input=summary_input,
            max_output_tokens=2000,
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


async def generate_message(
    llm: AsyncOpenAI,
    model_name: str,
    pg: PgClient,
    conversation_id: int,
    chat_meta: ChatMeta,
    max_tokens: int = 1024,
    lock: asyncio.Lock = None
) -> AsyncGenerator[str, None]:
    logger.info(
        "Requesting streamed LLM response (conversation_id=%s, model=%s, max_tokens=%s)",
        conversation_id,
        model_name,
        max_tokens,
    )
    try:
        stream = await llm.responses.create(
            model=model_name,
            instructions=SYSTEM_PROMPT,
            input=chat_meta.compaction,
            stream=True,
            max_output_tokens=max_tokens,
        )
        assistant_chunks: list[str] = []
        yield f"chat_id: {conversation_id}\n\n"
        async for event in stream:
            if event.type == "response.output_text.delta":
                assistant_chunks.append(event.delta)
                yield f"{event.delta}"
            elif event.type == "response.completed":
                break

        assistant_message = "".join(assistant_chunks)
        logger.info(
            "LLM response stream completed (conversation_id=%s, response_length=%s)",
            conversation_id,
            len(assistant_message),
        )
        chat_meta.messages.append({"role": "assistant", "content": assistant_message})
        chat_meta.compaction.append({"role": "assistant", "content": assistant_message})

        query = """
            UPDATE conversations
            SET
                messages = $1::jsonb,
                compaction = $2::jsonb
            WHERE id = $3
        """
        async with pg.transaction() as conn:
            await conn.execute(
                query,
                json.dumps(chat_meta.messages),
                json.dumps(chat_meta.compaction),
                conversation_id,
            )
        logger.info(
            "Conversation response persisted (conversation_id=%s)", conversation_id
        )
    except Exception:
        logger.exception(
            "Chat response generation failed (conversation_id=%s)", conversation_id
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
