from typing import AsyncGenerator

from google import genai
from google.genai import types

from app.core.log import get_logger
from app.core.prompts import IMAGE_DESCRIBE
from app.llm.llm_base import LLMBase

logger = get_logger(__name__)


class GoogleGenAILLM(LLMBase):
    def __init__(self, host: str | None = None, key: str | None = None):
        super().__init__(host, key)
        logger.info(f"Creating Google GenAI LLM client i.e gemini, host:{host}")

        # Initialize client with api_key; host can be passed via http_options if using custom endpoints
        http_options = types.HttpOptions(base_url=host) if host else None
        self.client = genai.Client(api_key=key, http_options=http_options)

    async def stream(
        self,
        model: str,
        input: list[dict],
        instructions: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        logger.info(f"Streaming Google GenAI response with model: {model}")

        # Build prompt contents and configuration
        contents = [
            types.Content(
                role=(
                    "model"
                    if msg.get("role") == "assistant"
                    else msg.get("role", "user")
                ),
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in input
        ]
        config = types.GenerateContentConfig(
            system_instruction=instructions,
            max_output_tokens=max_tokens,
        )

        try:
            # Use self.client.aio for async execution
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Google GenAI stream response failed: {e}")
            raise

    async def generate(
        self,
        model: str,
        input: list[dict],
        instructions: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ):
        logger.info("Generating Google GenAI response with model=%s", model)

        contents = [
            types.Content(
                role=(
                    "model"
                    if msg.get("role") == "assistant"
                    else msg.get("role", "user")
                ),
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in input
        ]
        config = types.GenerateContentConfig(
            system_instruction=instructions,
            max_output_tokens=max_tokens,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            logger.info(
                "Google GenAI response completed (usage=%r)",
                getattr(response, "usage_metadata", None),
            )
            return response.text

        except Exception as e:
            logger.error(f"Google GenAI generation failed: {e}")
            raise

    async def describe_image(
        self,
        image: bytes,
        mime_type: str,
        model: str,
    ) -> str:
        logger.info("Describing image with Google GenAI model=%s", model)

        try:
            # Pass image directly as Part bytes without needing base64 encoding
            image_part = types.Part.from_bytes(data=image, mime_type=mime_type)

            contents = [image_part, IMAGE_DESCRIBE]

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
            )

            return response.text

        except Exception as e:
            logger.error(f"Google GenAI image description failed (model={model}): {e}")
            raise
