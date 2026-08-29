from dataclasses import dataclass

from langchain_text_splitters import (MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)


@dataclass
class Splitters:
    markdown_splitter: MarkdownHeaderTextSplitter
    recursive_splitter: RecursiveCharacterTextSplitter


def create_splitters(chunk_size: int, chunk_overlap: int):
    md = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
        ],
        strip_headers=True,
    )
    rec = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splitters = Splitters(markdown_splitter=md, recursive_splitter=rec)
    return splitters
