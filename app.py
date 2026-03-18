"""
app.py — CLI entry point for the Autonomous Cognitive Engine.

Usage:
    python app.py
    python app.py --request "Research the latest trends in quantum computing"
    python app.py --verbose
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.table import Table

console = Console()


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]🤖 Autonomous Cognitive Engine[/bold cyan]\n"
            "[dim]Deep Research & Long-Horizon Task Framework[/dim]",
            border_style="cyan",
        )
    )


def print_todos(todos: list[dict]) -> None:
    if not todos:
        return
    table = Table(title="Task Plan", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Status", width=14)
    table.add_column("Description")

    icons = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
    for t in todos:
        icon = icons.get(t.get("status", "pending"), "⬜")
        table.add_row(t["id"], f"{icon} {t['status']}", t["description"])

    console.print(table)


def print_vfs(vfs: dict) -> None:
    if not vfs:
        return
    table = Table(title="Virtual File System", show_header=True, header_style="bold blue")
    table.add_column("Path", style="green")
    table.add_column("Size", justify="right")

    for path, entry in vfs.items():
        content = entry.get("content", "") if isinstance(entry, dict) else entry
        table.add_row(path, f"{len(content):,} chars")

    console.print(table)


def run(request: str, verbose: bool = False) -> str:
    """Execute the full agent pipeline for a given request."""
    from graph.build_graph import build_graph
    from memory.memory_manager import MemoryManager

    memory = MemoryManager()
    graph = build_graph()

    initial_state = {
        "messages": [],
        "todos": [],
        "virtual_fs": {},
        "current_task_id": None,
        "iteration": 0,
        "final_output": None,
        "user_request": request,
    }

    console.print(f"\n[bold green]Request:[/bold green] {request}\n")

    final_state = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Thinking…", total=None)

        for step in graph.stream(
            initial_state,
            stream_mode="values",
            config={"recursion_limit": 100},
        ):
            final_state = step

            if verbose:
                todos = step.get("todos", [])
                if todos:
                    progress.update(task, description=f"Working… ({sum(1 for t in todos if t['status'] == 'completed')}/{len(todos)} tasks done)")

    if final_state is None:
        console.print("[red]No output produced.[/red]")
        return ""

    # Print task summary
    todos = final_state.get("todos", [])
    if todos:
        print_todos(todos)

    # Print VFS summary
    vfs = final_state.get("virtual_fs", {})
    if vfs and verbose:
        print_vfs(vfs)

    # Print final output
    output = final_state.get("final_output", "")
    if output:
        console.print("\n")
        console.print(Panel(Markdown(output), title="[bold green]Final Output[/bold green]", border_style="green"))

        # Persist to episodic memory
        memory.after_run(request, output, todos)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Cognitive Engine")
    parser.add_argument(
        "--request", "-r",
        type=str,
        default="",
        help="Task request (if omitted, will prompt interactively)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress including VFS contents",
    )
    args = parser.parse_args()

    print_banner()

    request = args.request
    if not request:
        console.print("[bold yellow]Enter your request (Ctrl+D or empty line to finish):[/bold yellow]")
        lines = []
        try:
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
        except EOFError:
            pass
        request = "\n".join(lines).strip()

    if not request:
        console.print("[red]No request provided. Exiting.[/red]")
        sys.exit(1)

    try:
        run(request, verbose=args.verbose)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()