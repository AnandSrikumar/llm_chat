from io import BytesIO

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document as LangchainDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunkers.chunk_base import SyncChunker
from app.core.config import Settings
from app.core.log import get_logger

logger = get_logger(__name__)


class DocxChunker(SyncChunker):
    def __init__(self, settings: Settings):
        self.rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

    def _iter_block_items(self, parent):
        """
        Yield paragraphs and tables in their original document order.
        """
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        else:
            parent_elm = parent._tc

        for child in parent_elm.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, parent)
            elif child.tag.endswith("}tbl"):
                yield Table(child, parent)

    def _chunk_docx(
        self,
        file_data: bytes,
    ):
        try:
            doc = Document(BytesIO(file_data))

            parts = []

            for block in self._iter_block_items(doc):
                if isinstance(block, Paragraph):
                    text = block.text.strip()

                    if text:
                        parts.append(text)

                elif isinstance(block, Table):
                    rows = []

                    for row in block.rows:
                        cells = [cell.text.strip() for cell in row.cells]

                        if any(cells):
                            rows.append(" | ".join(cells))

                    if rows:
                        parts.append("\n".join(rows))

            text = "\n\n".join(parts)
            langchain_doc = LangchainDoc(
                page_content=text, metadata={"content_type": "docx"}
            )
            return self.rec_splitter.split_documents([langchain_doc])
        except Exception as e:
            logger.error(f"Docx parsing failed: {e}")
