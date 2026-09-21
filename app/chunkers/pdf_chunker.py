import pymupdf
import pymupdf4llm
from langchain_text_splitters import (MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)

from app.chunkers.chunk_base import SyncChunker
from app.core.config import Settings
from app.core.log import get_logger

logger = get_logger(__name__)


class PdfChunker(SyncChunker):
    def __init__(self, settings: Settings):
        self.rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
            ],
            strip_headers=True,
        )

    def chunk(self, data: bytes):
        doc = pymupdf.open(
            stream=data,
            filetype="pdf",
        )
        try:
            markdown = pymupdf4llm.to_markdown(doc)
            sections = self.md_splitter.split_text(markdown)
            chunks = self.rec_splitter.split_documents(sections)
            logger.info(
                "PDF chunking completed (sections=%s, chunks=%s)",
                len(sections),
                len(chunks),
            )
        except Exception:
            logger.exception("PDF chunking failed")
            raise
        finally:
            doc.close()
        return chunks
