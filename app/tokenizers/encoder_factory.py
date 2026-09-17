from fastapi.exceptions import HTTPException

from app.core.config import EncoderConfig
from app.core.log import get_logger
from app.tokenizers.encoders import Encoder, GeminiEncoder, HuggingFaceEncoder

logger = get_logger(__name__)


_TOKENIZER_MAP = {"google": {"encoder": GeminiEncoder, "is_api_key": True},
                  "huggingface": {"encoder": HuggingFaceEncoder, "is_api_key": False}}


def get_encoder(family: str, api_key: str, encoder_config: EncoderConfig):
    logger.info(f"Loading the encoder for chat model: {family}")
    encoder_type = encoder_config.type
    encoder_model = encoder_config.model

    encoder_meta = _TOKENIZER_MAP.get(encoder_type)
    if not encoder_meta:
        raise HTTPException(status_code=500, detail="encoding model failed to load")
    
    is_api_key = encoder_meta['is_api_key']
    encoder_obj: Encoder = encoder_meta['encoder']
    logger.info(f"Loading the encoder: {encoder_model}")

    if is_api_key:
        return encoder_obj(encoder_model, api_key)
    return encoder_obj(encoder_model)
