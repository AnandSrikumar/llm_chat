import base64

from openai import AsyncOpenAI

from app.core.prompts import IMAGE_DESCRIBE
from app.llm.llm_base import LLMBase

from app.core.log import get_logger

logger = get_logger(__name__)


class OpenAILLM(LLMBase):
    def __init__(self, host: str, key: str):
        super().__init__(host, key)
        logger.info(f"Creating OpenAI llm with {host}")
        self.client = AsyncOpenAI(base_url=host, api_key=key)

    async def stream(
        self,
        model: str,
        input: list[str],
        instructions: str = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ):
        logger.info(f"Streaming openai response with model: {model}")
        try:
            stream = await self.client.responses.create(
                model=model,
                input=input,
                instructions=instructions,
                max_output_tokens=max_tokens,
                stream=True,
                extra_body=extra_body,
            )
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

                elif event.type == "response.completed":
                    logger.info(
                        "status=%s incomplete_details=%r usage=%r",
                        event.response.status,
                        event.response.incomplete_details,
                        event.response.usage,
                    )
                    break

        except Exception as e:
            logger.error(f"OpenAI stream response failed: {e}")
            raise

    async def generate(
        self,
        model: int,
        input: list[str],
        instructions: str = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ):
        logger.info(
            "Generating OpenAI response with model=%s",
            model,
        )

        try:
            response = await self.client.responses.create(
                model=model,
                input=input,
                instructions=instructions,
                max_output_tokens=max_tokens,
                extra_body=extra_body,
            )

            logger.info(
                "OpenAI response completed " "(status=%s, usage=%r)",
                response.status,
                response.usage,
            )

            return response.output_text
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def describe_image(
        self,
        image: bytes,
        mime_type: str,
        model: str,
    ) -> str:

        logger.info(
            "Describing image with OpenAI model=%s",
            model,
        )

        try:
            encoded = base64.b64encode(image).decode("utf-8")

            response = await self.client.chat.completions.create(
                model=model,
                extra_body={"keep_alive": 0},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (f"data:{mime_type};" f"base64,{encoded}"),
                                },
                            },
                            {
                                "type": "text",
                                "text": IMAGE_DESCRIBE,
                            },
                        ],
                    }
                ],
            )

            return response.choices[0].message.content

        except Exception:
            logger.exception(
                "OpenAI image description failed " "(model=%s)",
                model,
            )
            raise
