from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from app.api.chat import router
from app.api.auth import router as auth_router
from app.core.config import Settings
from app.core.log import get_logger, initialize_logging, shutdown_logging
from app.core.pg_client import PgClient
from app.core.security import JWT, PasswordManager

logger = get_logger(__name__)


def create_app(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pg = None
        llm = None
        initialize_logging()
        logger.info("Starting LLM Chat application")
        try:
            pg = PgClient(settings.postgres_dsn, max_size=settings.pg_max_size)
            llm = AsyncOpenAI(
                base_url=f"{settings.ollama_host}/v1",
                api_key="ollama",
            )
            logger.info(
                "Connecting to PostgreSQL (pool_max_size=%s) and configuring LLM client (model=%s)",
                settings.pg_max_size,
                settings.ollama_chat_model,
            )
            await pg.connect()
            app.state.pg = pg
            app.state.llm = llm
            app.state.settings = settings
            pwd = PasswordManager()
            jwt = JWT(
                settings.SECRET_KEY,
                settings.ALGORITHM,
                settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            )
            app.state.pwd = pwd
            app.state.jwt = jwt
            app.state.tiktoken_encoding = AutoTokenizer.from_pretrained(
                settings.ollama_tokenizer
            )
            logger.info("LLM Chat application startup completed")
            yield
        except Exception:
            logger.exception("LLM Chat application startup failed")
            raise
        finally:
            try:
                if pg:
                    logger.info("Closing PostgreSQL connection pool")
                    await pg.close()
                logger.info("LLM Chat application shutdown completed")
            except Exception:
                logger.exception("LLM Chat application shutdown failed")
                raise
            finally:
                shutdown_logging()

    app = FastAPI(description="LLM Chat", lifespan=lifespan)
    app.include_router(router)
    app.include_router(auth_router)
    return app
