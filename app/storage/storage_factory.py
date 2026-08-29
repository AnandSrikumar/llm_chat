from app.storage.local_storage import LocalStorage
from app.storage.storage_base import Storage

STORAGE_MAP = {"local": LocalStorage}


def create_storage(storage_type: str, root: str) -> Storage:
    if storage_type not in STORAGE_MAP:
        raise AttributeError("Invalid storage")
    s_obj = STORAGE_MAP[storage_type](root)
    return s_obj
