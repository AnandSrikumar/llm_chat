from abc import ABC, abstractmethod


class LLMBase(ABC):
    def __init__(self, host: str, key: str):
        self.host = host
        self.key = key

    @abstractmethod
    async def stream(
        self,
        model: str,
        input: list[str],
        instructions: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ): ...

    @abstractmethod
    async def generate(
        self,
        model: str,
        input: list[str],
        instructions: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ): ...
