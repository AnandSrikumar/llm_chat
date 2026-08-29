from pathlib import Path

from app.core.log import get_logger
from app.storage.storage_base import Storage

logger = get_logger(__name__)


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
        logger.info(
            "Saving file to local storage (storage_key=%s, destination=%s, bytes=%s)",
            storage_key,
            path,
            len(file),
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(file)
        logger.info(
            "Local file write completed (storage_key=%s, destination=%s, bytes=%s)",
            storage_key,
            path,
            len(file),
        )

        return str(storage_key)

    async def delete_file(self, storage_key: str) -> None:
        path = self.base_path / storage_key

        if path.exists():
            logger.info(
                "Deleting local file (storage_key=%s, destination=%s)",
                storage_key,
                path,
            )
            path.unlink()
            logger.info("Local file deletion completed (storage_key=%s)", storage_key)
        else:
            logger.warning(
                "Local file deletion skipped because the file was not found (storage_key=%s, destination=%s)",
                storage_key,
                path,
            )
