import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
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
    max_tokens: int | None = 32000,
    chat_id: int | None = None,
    llm_model: str = "ministral-3:3b"
):
    model = llm[llm_model]
    res = await model.llm_object.generate(llm_model, [{"role": "user", "message":message}])
    
    return {"status": "ok", "message": res}
