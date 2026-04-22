import typer
from rich.console import Console

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
    console.print(f"[bold]Ingesting:[/bold] {path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about your documents"),
):
    """Ask a question about your ingested documents."""
    console.print(f"[bold]Question:[/bold] {question}")


@app.command()
def status():
    """Show what's been ingested."""
    console.print("[dim]No documents ingested yet.[/dim]")


if __name__ == "__main__":
    app()
