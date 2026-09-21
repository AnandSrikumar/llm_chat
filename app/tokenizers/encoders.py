from abc import ABC, abstractmethod

from google import genai
from transformers import AutoTokenizer

from app.core.log import get_logger

logger = get_logger(__name__)


class Encoder(ABC):

    @abstractmethod
    def encode(self, texts: list[str]): ...

    @abstractmethod
    def tokenize(self, texts: list[str]): ...

    @abstractmethod
    def count_tokens(self, texts: list[str]): ...


class HuggingFaceEncoder(Encoder):

    def __init__(self, model_name: str, tokenizer_model: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        logger.info(f"Huggingface encoder loaded: {model_name}")

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def tokenize(self, texts: list[str]):
        return [
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
            for text in texts
        ]

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(tokens) for tokens in self.tokenize(texts))


class GeminiEncoder(Encoder):

    def __init__(self, model: str, api_key: str, tokenizer: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tokenizer = tokenizer
        logger.info(f"Gemini encoder loaded")

    def encode(self, texts: list[str]):
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [embedding.values for embedding in response.embeddings]

    def tokenize(self, texts: list[str]):
        raise NotImplementedError(
            "Gemini tokenization is performed server-side. "
            "Use count_tokens() instead."
        )

    def count_tokens(self, texts: list[str]) -> int:
        response = self.client.models.count_tokens(
            model=self.model,
            contents=texts,
        )
        return response.total_tokens
