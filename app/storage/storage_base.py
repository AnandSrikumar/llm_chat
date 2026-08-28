from abc import ABC, abstractmethod


class Storage(ABC):

    @abstractmethod
    async def save_file(
        self,
        file: bytes,
        filename: str,
        owner_id: int,
        conversation_id: int,
    ) -> str:
        """Save file and return its storage key/path."""
        raise NotImplementedError

    @abstractmethod
    async def delete_file(self, storage_key: str) -> None:
        raise NotImplementedError
