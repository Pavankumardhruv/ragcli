from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragcli.answer import ask_question
from ragcli.embeddings import embed_texts
from ragcli.loader import load_path
from ragcli.store import VectorStore

app = typer.Typer(
    name="ragcli",
    help="Ask questions about your documents from the terminal.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to a file or directory to ingest"),
):
    """Ingest documents into the local vector store."""
    target = Path(path).expanduser().resolve()
    console.print(f"[bold]Ingesting:[/bold] {target}")

    with console.status("Loading and chunking documents..."):
        chunks = load_path(target)
    console.print(f"  Loaded [cyan]{len(chunks)}[/cyan] chunks from [cyan]{len({c.source for c in chunks})}[/cyan] files")

    with console.status("Generating embeddings (local model)..."):
        embeddings = embed_texts([c.text for c in chunks])

    store = VectorStore()
    count = store.add_chunks(chunks, embeddings)
    console.print(f"  Stored [green]{count}[/green] chunks in vector store")
    console.print("[bold green]Done.[/bold green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about your documents"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
):
    """Ask a question about your ingested documents."""
    store = VectorStore()
    if store.count() == 0:
        console.print("[red]No documents ingested yet.[/red] Run [bold]ragcli ingest <path>[/bold] first.")
        raise typer.Exit(1)

    console.print(f"[bold]Question:[/bold] {question}\n")

    with console.status("Searching documents and generating answer..."):
        response = ask_question(question, store, top_k=top_k)

    console.print(response)


@app.command()
def status():
    """Show what's been ingested."""
    store = VectorStore()
    count = store.count()

    if count == 0:
        console.print("[dim]No documents ingested yet.[/dim]")
        return

    sources = store.sources()
    table = Table(title="Ingested Documents")
    table.add_column("File", style="cyan")
    for src in sources:
        table.add_row(src)

    console.print(table)
    console.print(f"\n[bold]{count}[/bold] chunks across [bold]{len(sources)}[/bold] files")


@app.command()
def clear():
    """Clear all ingested documents."""
    import shutil
    from ragcli.store import DEFAULT_STORE_DIR

    if DEFAULT_STORE_DIR.exists():
        shutil.rmtree(DEFAULT_STORE_DIR)
        console.print("[yellow]Vector store cleared.[/yellow]")
    else:
        console.print("[dim]Nothing to clear.[/dim]")


if __name__ == "__main__":
    app()
