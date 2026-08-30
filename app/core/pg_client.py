from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pgvector.asyncpg import register_vector


import asyncpg

from app.core.log import get_logger

logger = get_logger(__name__)


class PgClient:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool: asyncpg.Pool | None = None

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    async def connect(self) -> None:
        if self.pool is not None:
            logger.info("PostgreSQL connection pool is already initialized")
            return

        logger.info(
            "Creating PostgreSQL connection pool (min_size=%s, max_size=%s)",
            self.min_size,
            self.max_size,
        )
        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            init=self._init_connection,
        )
        logger.info("PostgreSQL connection pool created")

    async def close(self) -> None:
        if self.pool is not None:
            logger.info("Closing PostgreSQL connection pool")
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")

    def _get_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            logger.error("PostgreSQL operation attempted before pool initialization")
            raise RuntimeError("PostgreSQL pool has not been initialized")

        return self.pool

    async def fetch(
        self,
        query: str,
        *args: Any,
    ) -> list[asyncpg.Record]:
        pool = self._get_pool()

        async with pool.acquire() as connection:
            logger.debug("Executing PostgreSQL fetch query")
            return await connection.fetch(query, *args)

    async def fetchone(
        self,
        query: str,
        *args: Any,
    ) -> asyncpg.Record | None:
        pool = self._get_pool()

        async with pool.acquire() as connection:
            logger.debug("Executing PostgreSQL fetch-one query")
            return await connection.fetchrow(query, *args)

    async def execute(
        self,
        query: str,
        *args: Any,
    ) -> str:
        pool = self._get_pool()

        async with pool.acquire() as connection:
            logger.debug("Executing PostgreSQL command")
            return await connection.execute(query, *args)

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[asyncpg.Connection]:
        pool = self._get_pool()

        async with pool.acquire() as connection:
            async with connection.transaction():
                logger.debug("PostgreSQL transaction started")
                try:
                    yield connection
                except Exception:
                    logger.exception("PostgreSQL transaction failed and will roll back")
                    raise
                else:
                    logger.debug("PostgreSQL transaction completed")
