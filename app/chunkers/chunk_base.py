from abc import ABC, abstractmethod

from app.core.splitters import Splitters


class SyncChunker(ABC):
    @abstractmethod
    def chunk(self, data: bytes): ...


class AsyncChunker(ABC):
    @abstractmethod
    async def chunk(self, data: bytes): ...
