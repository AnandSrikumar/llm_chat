from pathlib import Path

from app.storage.storage_base import Storage


class LocalStorage(Storage):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.storage_type = "local"

    async def save_file(
        self,
        file: bytes,
        filename: str,
        owner_id: int,
        conversation_id: int,
    ) -> str:

        storage_key = Path(str(owner_id)) / str(conversation_id) / filename

        path = self.base_path / storage_key

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(file)

        return str(storage_key)

    async def delete_file(self, storage_key: str) -> None:
        path = self.base_path / storage_key

        if path.exists():
            path.unlink()
