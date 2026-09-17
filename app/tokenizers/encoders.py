from abc import ABC, abstractmethod

from google import genai

from app.core.log import get_logger

logger = get_logger(__name__)


class Encoder(ABC):

    @abstractmethod
    def encode(self, texts: list[str]): ...


class HuggingFaceEncoder(Encoder):

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Huggingface encoder loaded")

    def encode(self, texts: list[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class GeminiEncoder(Encoder):

    def __init__(self, model: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        logger.info(f"Gemini encoder loaded")

    def encode(self, texts: list[str]):
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [embedding.values for embedding in response.embeddings]
