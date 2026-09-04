"""RAG internal data structures"""
from dataclasses import dataclass


@dataclass
class Chunk:
    """A retrievable text chunk: the unit stored in the vector store."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    tokens: int
    page: int | None = None  # txt has no page


@dataclass
class RetrievedChunk:
    """Return of retriever."""
    chunk: Chunk
    distance: float # cosine distance, lower = more similar