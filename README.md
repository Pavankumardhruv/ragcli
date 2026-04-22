<p align="center">
  <h1 align="center">ragcli</h1>
</p>

<p align="center">
  <strong>Ask questions about your documents from the terminal.</strong><br>
  Local embeddings. Claude-powered answers. No data leaves your machine (except the LLM call).
</p>

<p align="center">
  <a href="https://github.com/Pavankumardhruv/ragcli/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Pavankumardhruv/ragcli?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/embeddings-local-green?style=flat-square" alt="Local Embeddings">
  <img src="https://img.shields.io/badge/LLM-Claude-orange?style=flat-square" alt="Claude">
</p>

---

ragcli is a command-line tool that lets you ingest local documents (txt, md, pdf, py, json, csv) into a vector store, then ask natural language questions answered by Claude using only your documents as context.

Embeddings run 100% locally via [sentence-transformers](https://www.sbert.net/). Only the final question + retrieved context is sent to the Claude API for answer generation.

## Quick Start

```bash
# Install
pip install -e .

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here

# Ingest a directory of documents
ragcli ingest ./docs/

# Ask a question
ragcli ask "What is the refund policy?"

# See what's been ingested
ragcli status
```

## How It Works

```
                    ┌──────────────┐
  Your Documents ──►│  Chunking    │  Split into overlapping 512-word chunks
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Embeddings  │  all-MiniLM-L6-v2 (runs locally)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  ChromaDB    │  Persistent local vector store
                    └──────┬───────┘
                           │
  Your Question ──────────►│  Cosine similarity search
                           │
                    ┌──────▼───────┐
                    │  Claude API  │  Generates grounded answer with citations
                    └──────────────┘
```

## Commands

| Command | Description |
|---|---|
| `ragcli ingest <path>` | Ingest a file or directory into the vector store |
| `ragcli ask "<question>"` | Ask a question about your ingested documents |
| `ragcli status` | Show all ingested files and chunk count |
| `ragcli clear` | Delete all ingested data |

## Supported File Types

- `.txt` — Plain text
- `.md` — Markdown
- `.pdf` — PDF documents (via pypdf)
- `.py` — Python source code
- `.json` — JSON files
- `.csv` — CSV files

## Architecture

```
ragcli/
├── cli.py          # Typer CLI commands
├── loader.py       # File reading + text chunking
├── embeddings.py   # Local sentence-transformer embeddings
├── store.py        # ChromaDB vector store wrapper
└── answer.py       # Claude API integration for answer generation
```

**Design decisions:**

- **Local embeddings** — Your documents never leave your machine during ingestion. The embedding model (all-MiniLM-L6-v2, ~80MB) runs entirely on-device.
- **Overlapping chunks** — 512-word chunks with 64-word overlap prevent context loss at chunk boundaries.
- **Persistent store** — ChromaDB persists to `~/.ragcli/store/`, so you ingest once and query many times.
- **Upsert on re-ingestion** — Re-ingesting the same file updates existing chunks instead of duplicating them.

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) for answer generation

## Installation

```bash
git clone https://github.com/Pavankumardhruv/ragcli.git
cd ragcli
pip install -e .
```

## License

MIT License — see [LICENSE](LICENSE) for details.
