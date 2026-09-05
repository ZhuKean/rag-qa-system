"""Document loaders: convert raw files into page-level plain text."""

from pypdf import PdfReader
from docx import Document

def load_text(path: str) -> list[str]:
    """Load .txt/.md: return [whole file text]."""
    if not path.endswith((".txt", ".md")):
        raise ValueError("Path must end with .txt/.md")
    with open(path, "r", encoding="utf-8") as f:
        return [f.read()]

def load_docx(path: str) -> list[str]:
    """Load .docx: return [whole file text].  Using python-docx"""
    if not path.endswith(".docx"):
        raise ValueError("Path must end with .docx")
    document = Document(path)
    text = "\n\n".join(p.text for p in document.paragraphs)
    return [text]

def load_pdf(path: str) -> list[str]:
    """Load .pdf: return [page1_text, page2_text, ...].  Using pypdf"""
    if not path.endswith(".pdf"):
        raise ValueError("Path must end with .pdf")
    reader = PdfReader(path)
    return [page.extract_text() for page in reader.pages]