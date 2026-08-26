import re
from langchain_core.documents import Document


def clean_text_for_bm25(text: str) -> str:
    # Lowercase
    text = text.lower()

    # Remove code fences
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove markdown headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown links but preserve the text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove emphasis markers
    text = re.sub(r"[*_~]", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep words/numbers, replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_chunks_for_bm25(chunks: list[Document]) -> list[Document]:
    for chunk in chunks:
        chunk.metadata["bm25_text"] = clean_text_for_bm25(
            chunk.page_content
        )

    return chunks