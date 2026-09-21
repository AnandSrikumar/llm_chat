from charset_normalizer import from_bytes
from langchain_core.documents import Document as LangchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunkers.chunk_base import SyncChunker
from app.core.config import Settings


class TextChunker(SyncChunker):
    def __init__(self, settings: Settings):
        self.rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

    def chunk(self, data: bytes):
        text = str(from_bytes(data).best())
        doc = LangchainDoc(text)
        return self.rec_splitter.split_documents([doc])
