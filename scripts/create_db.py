import argparse
import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


load_dotenv()

def build_dsn() -> str:
    host = os.environ["POSTGRES_HOST"]
    port = os.environ["POSTGRES_PORT"]
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    database = os.environ["POSTGRES_DB"]

    return (
        f"postgresql://{user}:{password}"
        f"@{host}:{port}/{database}"
    )

async def init_db(schema_path: Path) -> None:
    database_url = build_dsn()

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    schema_sql = schema_path.read_text(encoding="utf-8")

    pool = await asyncpg.create_pool(dsn=database_url)

    try:
        async with pool.acquire() as connection:
            await connection.execute(schema_sql)

        print(f"Database initialized successfully using: {schema_path}")

    finally:
        await pool.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL database using a SQL schema file."
    )

    parser.add_argument(
        "schema",
        type=Path,
        help="Path to the SQL schema file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.schema.is_file():
        raise FileNotFoundError(f"Schema file not found: {args.schema}")

    asyncio.run(init_db(args.schema))


if __name__ == "__main__":
    main()