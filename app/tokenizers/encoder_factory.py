from fastapi.exceptions import HTTPException

from app.core.log import get_logger
from app.tokenizers.encoders import Encoder, GeminiEncoder, HuggingFaceEncoder

logger = get_logger(__name__)

_TOKENIZER_MAP = {
    "qwen3:4b": {
        "model": "Qwen/Qwen3-4B",
        "encoder": HuggingFaceEncoder,
        "api_key": False,
    },
    "ministral-3:3b": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512",
        "encoder": HuggingFaceEncoder,
        "api_key": False,
    },
    "gemini-2.5-flash": {
        "model": "gemini-embedding-2",
        "encoder": GeminiEncoder,
        "api_key": True,
    },
    "gemini-3.8-flash": {
            "model": "gemini-embedding-2",
            "encoder": GeminiEncoder,
            "api_key": True,
        },
}


def get_encoder(model_name: str, api_key: str):
    logger.info(f"Loading the encoder for chat model: {model_name}")
    model_encoder_meta: dict = _TOKENIZER_MAP.get(model_name)

    if not model_encoder_meta:
        raise HTTPException(status_code=500, detail="encoding model failed to load")

    encoder = model_encoder_meta["encoder"]
    is_api_key = model_encoder_meta["api_key"]
    model_encoder = model_encoder_meta["model"]

    logger.info(f"Loading the encoder: {model_encoder}")

    if is_api_key:
        return encoder(model_encoder, api_key)
    return encoder(model_encoder)
