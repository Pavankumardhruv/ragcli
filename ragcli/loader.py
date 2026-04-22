from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def load_path(path: Path) -> list[Chunk]:
    path = Path(path)
    files = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [f for f in path.rglob("*") if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not files:
        raise ValueError(f"No supported files found in {path}")

    chunks = []
    for file in sorted(files):
        text = read_file(file)
        if not text.strip():
            continue
        for i, chunk_text_str in enumerate(chunk_text(text)):
            chunks.append(Chunk(text=chunk_text_str, source=str(file), chunk_index=i))

    return chunks
