from dataclasses import dataclass

from app.core.config import LLMModelConfig, Settings
from app.llm.llm_base import LLMBase
from app.llm.openai_llm import OpenAILLM
from app.tokenizers.encoder_factory import get_encoder
from app.tokenizers.encoders import Encoder

from app.core.log import get_logger
logger = get_logger(__name__)

_LLM_FAMILY_MAP = {"ministral": OpenAILLM}


@dataclass
class LLMModelObject:    
    llm_object: LLMBase
    encoding_object: Encoder
    
def create_llm_object(settings: Settings):
    model_map = {}
    models: dict[str, LLMModelConfig] = settings.models
    logger.info(f"The models are: {models}")
    for _, model in models.items():
        family = model.family
        logger.info(f"Loading the LLM model and encoder for: {family}")
        if family not in _LLM_FAMILY_MAP:
            continue
        api_key = getattr(settings, model.api_key_env)
        llm_obj = _LLM_FAMILY_MAP[family](model.host, api_key)
        
        encoder = get_encoder(family, api_key, model.encoder)
        for varient in model.variants:
            model_map[varient] = LLMModelObject(llm_object=llm_obj, encoding_object=encoder)
    return model_map