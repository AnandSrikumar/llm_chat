from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_TOKENIZER_MAP = {
    "qwen3:4b": "Qwen/Qwen3-4B",
    "ministral:3b": "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral-3:3b": "mistralai/Ministral-3-3B-Instruct-2512",
}


class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    pg_max_size: int

    ollama_host: str
    ollama_chat_model: str = "qwen3:4b"

    chunk_type: str
    chunk_size: int
    chunk_overlap: int

    embedding_model: str
    embedding_dims: int

    compact_threshold: int = 15000

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

    @computed_field
    @property
    def ollama_tokenizer(self) -> str:
        try:
            return _TOKENIZER_MAP[self.ollama_chat_model]
        except KeyError:
            raise ValueError(
                f"No tokenizer configured for Ollama model: "
                f"{self.ollama_chat_model}"
            )
