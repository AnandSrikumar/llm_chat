from typing import Optional

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class EncoderConfig(BaseModel):
    type: str
    model: str
    tokenizer: Optional[str] = None


class LLMModelConfig(BaseModel):
    family: str
    variants: list[str]
    host: str
    encoder: EncoderConfig
    api_key_env: str
    vision_model: str | None = None


class Settings(BaseSettings):
    # PostgreSQL
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    pg_max_size: int

    # API keys
    gemini_api_key: str
    ministral_api_key: str

    # LLM model registry
    models: dict[str, LLMModelConfig]

    # Chunking
    chunk_type: str
    chunk_size: int
    chunk_overlap: int

    # Application
    compact_threshold: int = 20000
    max_tokens: int | None = None

    # Storage
    storage_type: str = "local"
    storage_root: str = "./data"

    # Authentication
    SECRET_KEY: str = "EGlS6s24wUVdfXjVkh3U5Yktw9brjEIFWD5nRgK2KXk"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            "llm_models.yml",
        )

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings,
        )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )
