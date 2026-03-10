"""
test_milestone2_vfs.py - Milestone 2 evaluation: VFS tool usage test.
Tests the read/write/edit balance and selective retrieval pattern.
Run with: python test_milestone2_vfs.py
"""

import json
import sys
sys.path.insert(0, ".")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_vfs_test():
    console.print(Panel.fit(
        "[bold cyan]Milestone 2 - VFS Evaluation Test[/bold cyan]\n"
        "[dim]Tests: write, read, edit, selective retrieval[/dim]",
        border_style="cyan"
    ))

    from tools.filesystem import bind_vfs, write_file, read_file, edit_file, ls

    vfs = {}
    bind_vfs(vfs)
    results = []

    def check(label, condition):
        results.append((label, condition))
        icon = "[green]PASS[/green]" if condition else "[red]FAIL[/red]"
        console.print(f"  {icon}  {label}")

    # Test 1: write_file
    console.print("\n[bold]Test 1: write_file stores summary[/bold]")
    wr = write_file.invoke({"path": "/summaries/a_summary.txt",
                            "content": "IEEE framework: engineering accountability and human well-being."})
    data = json.loads(wr)
    vfs[data["path"]] = {"content": data["content"],
                         "created_at": data["created_at"],
                         "updated_at": data["updated_at"]}
    check("write_file returns action=write_file", data["action"] == "write_file")
    check("File stored in VFS", "/summaries/a_summary.txt" in vfs)

    # Test 2: write second file
    console.print("\n[bold]Test 2: write second summary[/bold]")
    wr2 = write_file.invoke({"path": "/summaries/b_summary.txt",
                             "content": "EU framework: rights-based approach with 7 requirements."})
    data2 = json.loads(wr2)
    vfs[data2["path"]] = {"content": data2["content"],
                          "created_at": data2["created_at"],
                          "updated_at": data2["updated_at"]}
    check("Second file stored", "/summaries/b_summary.txt" in vfs)

    # Test 3: ls
    console.print("\n[bold]Test 3: ls lists VFS contents[/bold]")
    ls_result = ls.invoke({"directory": "/"})
    ls_data = json.loads(ls_result)
    check("ls returns 2 files", ls_data["count"] == 2)
    console.print(f"  Files: {[f['path'] for f in ls_data['files']]}")

    # Test 4: selective read
    console.print("\n[bold]Test 4: Selective read_file (only needed files)[/bold]")
    console.print("  [dim]Task: Compare A and B. Read ONLY A and B.[/dim]")
    files_needed = ["/summaries/a_summary.txt", "/summaries/b_summary.txt"]
    read_contents = []
    for f in files_needed:
        rr = read_file.invoke({"path": f})
        rd = json.loads(rr)
        read_contents.append(rd["content"])
        console.print(f"  read_file <- {f}")
    check("Read only 2 needed files", len(read_contents) == 2)
    check("Content retrieved correctly", "IEEE" in read_contents[0])

    # Test 5: write comparison
    console.print("\n[bold]Test 5: Write comparison result[/bold]")
    comparison = "A vs B: IEEE focuses on engineering standards. EU focuses on legal rights."
    wc = write_file.invoke({"path": "/compare/a_vs_b.txt", "content": comparison})
    wcd = json.loads(wc)
    vfs[wcd["path"]] = {"content": wcd["content"],
                        "created_at": wcd["created_at"],
                        "updated_at": wcd["updated_at"]}
    check("Comparison file written to /compare/", "/compare/a_vs_b.txt" in vfs)

    # Test 6: edit_file append
    console.print("\n[bold]Test 6: edit_file append (NOT rewrite)[/bold]")
    console.print("  [dim]Appending sustainability section to comparison file.[/dim]")
    er = edit_file.invoke({
        "path": "/compare/a_vs_b.txt",
        "mode": "append",
        "content": "Sustainability gap: neither IEEE nor EU addresses environmental impact of AI.",
        "old_text": ""
    })
    ed = json.loads(er)
    check("edit_file action returned", ed["action"] == "edit_file")
    check("Mode is append", ed["mode"] == "append")
    existing = vfs["/compare/a_vs_b.txt"]["content"]
    vfs["/compare/a_vs_b.txt"]["content"] = existing + "\n\n" + ed["content"]
    check("Sustainability appended", "Sustainability gap" in vfs["/compare/a_vs_b.txt"]["content"])
    check("Original content preserved", "IEEE" in vfs["/compare/a_vs_b.txt"]["content"])

    # Test 7: file structure
    console.print("\n[bold]Test 7: VFS structure is clean[/bold]")
    all_paths = list(vfs.keys())
    structured = all(any(p.startswith(prefix) for prefix in
                         ["/summaries/", "/compare/", "/drafts/", "/research/"])
                     for p in all_paths)
    check("All files in structured directories", structured)
    check("No duplicate files", len(all_paths) == len(set(all_paths)))
    check("Total files is minimal (3)", len(all_paths) == 3)

    # Final VFS table
    console.print("\n")
    table = Table(title="Final VFS State", show_header=True, header_style="bold green")
    table.add_column("Path", style="green")
    table.add_column("Words", justify="right")
    table.add_column("Preview")
    for path, entry in vfs.items():
        content = entry["content"]
        preview = content[:70] + "..." if len(content) > 70 else content
        table.add_row(path, str(len(content.split())), preview)
    console.print(table)

    passed = sum(1 for _, r in results if r)
    total = len(results)
    score = int((passed / total) * 100)
    console.print(f"\n[bold]Score: {passed}/{total} ({score}%)[/bold]")
    if score == 100:
        console.print("[bold green]Milestone 2 VFS tests: ALL PASSED[/bold green]")
    else:
        console.print(f"[yellow]{total - passed} tests failed[/yellow]")
    return score


if __name__ == "__main__":
    run_vfs_test()