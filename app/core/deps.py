from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from tiktoken import Encoding

from app.core.log import get_logger
from app.core.pg_client import PgClient
from app.core.security import JWT, PasswordManager

logger = get_logger(__name__)


def get_pg(request: Request) -> PgClient:
    return request.app.state.pg


def get_llm(request: Request) -> AsyncOpenAI:
    return request.app.state.llm


def get_llm_model(request: Request) -> str:
    return request.app.state.settings.ollama_chat_model


def get_compact_thres(request: Request) -> int:
    return request.app.state.settings.compact_threshold


bearer = HTTPBearer()


async def get_current_user(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials,
        Depends(bearer),
    ],
):
    security: JWT = request.app.state.jwt

    try:
        payload = security.decode_access_token(credentials.credentials)

    except jwt.ExpiredSignatureError:
        logger.info("Authentication rejected: expired access token")
        raise HTTPException(
            status_code=401,
            detail="Token expired",
        )

    except jwt.InvalidTokenError:
        logger.info("Authentication rejected: invalid access token")
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    username = payload.get("username")
    uid = payload.get("id")

    if not username:
        logger.warning("Authentication rejected: token is missing a username claim")
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    logger.info("Authentication succeeded for user_id=%s", uid)
    return {"username": username, "id": uid}


def get_password_manager(request: Request):
    return request.app.state.pwd


def get_jwt(request: Request):
    return request.app.state.jwt


def get_tiktoken(request: Request):
    return request.app.state.tiktoken_encoding


Pg = Annotated[PgClient, Depends(get_pg)]
LLM = Annotated[AsyncOpenAI, Depends(get_llm)]
LLM_MODEL = Annotated[str, Depends(get_llm_model)]
USER = Annotated[dict, Depends(get_current_user)]
COMPACT_THRESHOLD = Annotated[int, Depends(get_compact_thres)]
JWT_DEP = Annotated[JWT, Depends(get_jwt)]
PASSWORD_MANAGER = Annotated[PasswordManager, Depends(get_password_manager)]
TIKTOKEN_ENCODING = Annotated[Encoding, Depends(get_tiktoken)]
