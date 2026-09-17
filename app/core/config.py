from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    pg_max_size: int

    ollama_host: str
    ollama_chat_model: str = "qwen3:4b"
    ollama_vision_model: str = "qwen2.5vl:3b"
    ollama_key: str
    ollama_key_vision: str
    chunk_type: str
    chunk_size: int
    chunk_overlap: int

    embedding_model: str
    embedding_dims: int

    compact_threshold: int = 20000

    storage_type: str = "local"
    storage_root: str = "./data"

    SECRET_KEY: str = "EGlS6s24wUVdfXjVkh3U5Yktw9brjEIFWD5nRgK2KXk"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )
