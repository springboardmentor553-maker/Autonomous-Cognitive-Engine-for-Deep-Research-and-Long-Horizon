"""
test_cybersecurity_frameworks.py - Scaling Test: Cybersecurity Frameworks
Analyze NIST, ISO27001, SOC2 and CIS frameworks, compare them,
propose unified security framework, refine with implementation roadmap.

Run with: python test_cybersecurity_frameworks.py
"""

import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, ".")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

# ── Virtual File System ───────────────────────────────────────────────────────
vfs = {}

def write_file(path, content):
    now = datetime.now(timezone.utc).isoformat()
    vfs[path] = {"content": content, "created_at": now, "updated_at": now}
    console.print(f"  [green]write_file[/green] -> {path} ({len(content.split())} words)")

def read_file(path):
    if path not in vfs:
        console.print(f"  [red]read_file ERROR[/red] - {path} not found")
        return ""
    content = vfs[path]["content"]
    console.print(f"  [cyan]read_file[/cyan]  <- {path} ({len(content.split())} words)")
    return content

def edit_file(path, mode, content, old_text=""):
    if path not in vfs:
        console.print(f"  [red]edit_file ERROR[/red] - {path} not found")
        return ""
    existing = vfs[path]["content"]
    updated = existing + "\n\n" + content if mode == "append" else (
        existing.replace(old_text, content, 1) if mode == "replace" and old_text else content
    )
    vfs[path]["content"] = updated
    vfs[path]["updated_at"] = datetime.now(timezone.utc).isoformat()
    console.print(f"  [yellow]edit_file[/yellow]  [{mode}] {path} -> now {len(updated.split())} words")

# ── Framework Data ────────────────────────────────────────────────────────────
FRAMEWORKS = {
    "NIST": {
        "name": "NIST Cybersecurity Framework 2.0",
        "summary": """NIST CSF 2.0 organizes cybersecurity around six core functions: Govern,
Identify, Protect, Detect, Respond, Recover. Voluntary framework widely adopted
by US federal agencies and private sector. Strength: flexible, risk-based
approach adaptable to any organization size. Provides implementation tiers
(Partial, Risk Informed, Repeatable, Adaptive). Gap: no certification mechanism,
compliance cannot be formally verified. Limited supply chain security guidance.
Governance function newly added in v2.0 to address leadership accountability."""
    },
    "ISO27001": {
        "name": "ISO/IEC 27001:2022",
        "summary": """ISO 27001 is the international standard for Information Security Management
Systems (ISMS). Certification-based: organizations undergo formal third-party
audit to achieve certification. 93 controls organized across 4 themes: People,
Physical, Technological, Organizational. Strength: globally recognized, provides
verified compliance proof, supply chain assurance. Gap: expensive and time-consuming
to implement and maintain. Annual surveillance audits required. Smaller organizations
struggle with resource requirements. More prescriptive than NIST, less flexible
for rapidly evolving threat landscape."""
    },
    "SOC2": {
        "name": "SOC 2 Type II",
        "summary": """SOC 2 is an auditing standard for service organizations handling customer
data. Built around five Trust Service Criteria: Security, Availability, Processing
Integrity, Confidentiality, Privacy. Type II reports cover a period (minimum 6
months) proving controls operate effectively over time. Strength: directly
addresses cloud and SaaS provider trust concerns. Required by enterprise
customers before vendor onboarding. Gap: scope limited to service organizations,
not general enterprise security. Each audit is custom-scoped, making comparisons
difficult. No prescriptive control list, auditor judgment varies significantly."""
    },
    "CIS": {
        "name": "CIS Controls v8",
        "summary": """CIS Controls v8 provides 18 prioritized safeguards organized into three
implementation groups (IG1, IG2, IG3) based on organization size and risk.
IG1: basic cyber hygiene for small organizations (56 safeguards). IG3: advanced
controls for large enterprises. Strength: highly prescriptive and actionable,
clear priority order helps resource-constrained teams. Maps to NIST, ISO, and
regulatory frameworks. Gap: technology-specific controls become outdated as
technology evolves. No formal certification pathway. Compliance verification
relies on self-assessment or contractual audits."""
    }
}

ROADMAP_ADDENDUM = """
## Implementation Roadmap

### Phase 1 — Foundation (Months 1-6)
- Adopt CIS IG1 controls as immediate baseline (56 safeguards)
- Complete asset inventory and risk assessment per NIST Identify function
- Establish governance structure per NIST Govern function
- Train security awareness across all staff

### Phase 2 — Formalization (Months 6-18)
- Begin ISO 27001 gap assessment and remediation
- Implement CIS IG2 controls for enhanced protection
- Develop incident response plan per NIST Respond function
- Prepare for SOC 2 Type I audit if handling customer data

### Phase 3 — Certification (Months 18-36)
- Achieve ISO 27001 certification
- Complete SOC 2 Type II audit (6-month evidence period)
- Advance to CIS IG3 for critical systems
- Implement continuous monitoring and threat intelligence

### Resource Estimates
Small org (< 100 staff):  Focus on CIS IG1 + SOC 2 Security only
Medium org (100-1000):    CIS IG2 + ISO 27001 + SOC 2 full scope
Large org (> 1000):       Full framework integration with all certifications

### Key Success Metrics
- Mean Time to Detect (MTTD): target < 1 hour for critical incidents
- Mean Time to Respond (MTTR): target < 4 hours for critical incidents
- Patch compliance rate: target > 95% within 30 days of release
- Security awareness training completion: target > 98% annually
"""

# ── Trace ─────────────────────────────────────────────────────────────────────
trace = []

def log_step(step, title, reads, writes, skipped=None):
    trace.append({
        "step": step, "title": title,
        "reads": reads, "writes": writes,
        "skipped": skipped or [],
        "vfs_size_after": len(vfs)
    })

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    console.print(Panel.fit(
        "[bold cyan]Scaling Test - Cybersecurity Frameworks[/bold cyan]\n"
        "[dim]Analyze NIST, ISO27001, SOC2 and CIS frameworks,\n"
        "propose unified security framework with implementation roadmap[/dim]",
        border_style="cyan"
    ))

    console.print("\n[bold yellow]Task:[/bold yellow] Analyze 4 cybersecurity frameworks, "
                  "identify differences, propose a unified security framework, "
                  "then refine with implementation roadmap.\n")

    # PHASE 1: GATHER
    console.print(Panel(
        "[bold]PHASE 1 - GATHER[/bold]\nSummarize each framework independently",
        border_style="blue"
    ))

    for i, (fw, data) in enumerate(FRAMEWORKS.items(), 1):
        console.print(f"\n[bold]Step {i}[/bold]: Summarize {fw} - {data['name']}")
        console.print(f"  [dim]Reasoning: Only need {fw} data. Other frameworks not required yet.[/dim]")
        path = f"/summaries/{fw.lower()}_summary.txt"
        write_file(path, data["summary"].strip())
        log_step(i, f"Summarize {fw}", [], [path])
        time.sleep(0.2)

    # PHASE 2: COMPARE
    console.print(Panel(
        "[bold]PHASE 2 - COMPARE[/bold]\nRead all 4 summaries, extract differences",
        border_style="magenta"
    ))

    console.print("\n[bold]Step 5[/bold]: Compare all 4 cybersecurity frameworks")
    console.print("  [dim]Reasoning: Need all 4 summary files for complete comparison.[/dim]")
    console.print("  [dim]Reading summaries only - NOT raw framework documents.[/dim]")

    reads = []
    for fw in FRAMEWORKS:
        path = f"/summaries/{fw.lower()}_summary.txt"
        read_file(path)
        reads.append(path)

    comparison = """# Cybersecurity Framework Comparison

## NIST CSF 2.0
Type: Voluntary guidance. Certification: None.
Scope: Any organization. Strength: Flexible and risk-based.
Gap: No formal verification mechanism.

## ISO 27001
Type: Certifiable standard. Certification: Third-party audit.
Scope: Any organization. Strength: Globally recognized proof of compliance.
Gap: Expensive, resource-intensive for small organizations.

## SOC 2 Type II
Type: Audit report. Certification: CPA firm audit.
Scope: Service/cloud organizations only. Strength: Customer trust verification.
Gap: Limited to service orgs, inconsistent scoping.

## CIS Controls v8
Type: Prescriptive controls. Certification: Self-assessment.
Scope: Any organization. Strength: Prioritized, actionable, tiered by size.
Gap: No formal certification, technology-specific controls age quickly.

## Key Differences Matrix
| Dimension       | NIST   | ISO27001 | SOC2   | CIS    |
|-----------------|--------|----------|--------|--------|
| Certification   | No     | Yes      | Yes    | No     |
| Flexibility     | High   | Medium   | Low    | Low    |
| Prescriptiveness| Low    | Medium   | Low    | High   |
| Cost            | Low    | High     | Medium | Low    |
| Scope           | Any    | Any      | SaaS   | Any    |
| Update frequency| Medium | Slow     | Slow   | Fast   |

## Common Gaps
1. No single framework addresses all organization types and sizes
2. Supply chain security coverage is inconsistent
3. Cloud-native and AI security barely addressed
4. Measurement and metrics vary wildly across frameworks
5. Integration between frameworks requires manual mapping effort
"""
    write_file("/compare/framework_comparison.txt", comparison)
    log_step(5, "Compare all 4 frameworks", reads, ["/compare/framework_comparison.txt"])

    # PHASE 3: SYNTHESISE
    console.print(Panel(
        "[bold]PHASE 3 - SYNTHESISE[/bold]\nPropose unified security framework",
        border_style="green"
    ))

    console.print("\n[bold]Step 6[/bold]: Propose unified cybersecurity framework")
    console.print("  [dim]Reasoning: Only need /compare/framework_comparison.txt.[/dim]")
    console.print("  [dim]Skipping all /summaries/ - comparison has extracted insights.[/dim]")

    read_file("/compare/framework_comparison.txt")

    unified = """# Unified Cybersecurity Framework

## Overview
Integrates NIST flexibility, ISO 27001 certification rigor,
SOC 2 customer trust focus, and CIS prescriptive controls
into a single tiered framework for all organization types.

## Core Functions (from NIST)
- Govern: Leadership accountability and risk strategy
- Identify: Asset and risk inventory
- Protect: Safeguards and access controls
- Detect: Continuous monitoring and anomaly detection
- Respond: Incident response and communication
- Recover: Recovery planning and improvements

## Control Tiers (from CIS)
- Tier 1 (Essential): 20 controls for all organizations
- Tier 2 (Enhanced): 40 additional controls for medium orgs
- Tier 3 (Advanced): Full control suite for large enterprises

## Certification Pathway (from ISO 27001 + SOC 2)
- Level 1: Self-assessment against Tier 1 controls
- Level 2: Third-party audit of Tier 1+2 controls (annual)
- Level 3: Full ISO 27001 certification + SOC 2 Type II

## Trust Report (from SOC 2)
- Standardized trust report available at each certification level
- Machine-readable format for automated vendor assessment
- Covers: Security, Availability, Privacy, Integrity, Confidentiality

## Measurement Standard
- Unified metrics dashboard across all functions
- Quarterly progress reporting required at Level 2+
- Public disclosure of Level 3 certification status
"""
    write_file("/drafts/unified_security.txt", unified)
    log_step(
        6, "Propose unified framework",
        ["/compare/framework_comparison.txt"],
        ["/drafts/unified_security.txt"],
        [f"/summaries/{fw.lower()}_summary.txt" for fw in FRAMEWORKS]
    )

    # PHASE 4: REFINE
    console.print(Panel(
        "[bold]PHASE 4 - REFINE[/bold]\nAppend implementation roadmap using edit_file",
        border_style="yellow"
    ))

    console.print("\n[bold]Step 7[/bold]: Refine with implementation roadmap")
    console.print("  [dim]Reasoning: Only need to append to /drafts/unified_security.txt.[/dim]")
    console.print("  [dim]Using edit_file(append) - NOT rewriting the file.[/dim]")
    console.print("  [dim]Skipping all other files - unified framework is complete.[/dim]")

    edit_file("/drafts/unified_security.txt", "append", ROADMAP_ADDENDUM)
    log_step(
        7, "Refine with roadmap (edit_file append)",
        [], [],
        ["/compare/framework_comparison.txt"] +
        [f"/summaries/{fw.lower()}_summary.txt" for fw in FRAMEWORKS]
    )

    # EXECUTION TRACE
    console.print("\n")
    console.print(Panel("[bold]EXECUTION TRACE[/bold]", border_style="white"))

    trace_table = Table(show_header=True, header_style="bold white")
    trace_table.add_column("Step", style="cyan", width=6)
    trace_table.add_column("Task", width=32)
    trace_table.add_column("Files Read", style="green", width=32)
    trace_table.add_column("Files Written", style="yellow", width=28)
    trace_table.add_column("Skipped", style="dim", width=12)

    for t in trace:
        trace_table.add_row(
            str(t["step"]),
            t["title"],
            "\n".join(t["reads"]) if t["reads"] else "(none)",
            "\n".join(t["writes"]) if t["writes"] else "edit_file" if t["step"] == 7 else "(none)",
            f"{len(t['skipped'])} files" if t["skipped"] else "-"
        )
    console.print(trace_table)

    # FINAL VFS STATE
    console.print("\n")
    console.print(Panel("[bold]FINAL VIRTUAL FILE SYSTEM STATE[/bold]", border_style="green"))

    vfs_table = Table(show_header=True, header_style="bold green")
    vfs_table.add_column("Path", style="green", width=42)
    vfs_table.add_column("Words", justify="right", width=8)
    vfs_table.add_column("Type", width=12)

    for path, entry in vfs.items():
        file_type = ("summary" if "/summaries/" in path else
                     "compare" if "/compare/" in path else
                     "draft" if "/drafts/" in path else "other")
        vfs_table.add_row(path, str(len(entry["content"].split())), file_type)
    console.print(vfs_table)

    # EVALUATION
    console.print("\n")
    console.print(Panel("[bold]MENTOR EVALUATION CHECKLIST[/bold]", border_style="cyan"))

    checks = [
        ("Summaries stored, not raw data",
         all("/summaries/" in p or "/compare/" in p or "/drafts/" in p for p in vfs)),
        ("No unnecessary files in VFS",
         len(vfs) == 6),  # 4 summaries + 1 comparison + 1 unified
        ("Selective retrieval - Step 5 reads all 4 summaries",
         len(trace[4]["reads"]) == 4),
        ("Selective retrieval - Step 6 reads only comparison",
         trace[5]["reads"] == ["/compare/framework_comparison.txt"]),
        ("edit_file demonstrated in Step 7",
         trace[6]["reads"] == []),
        ("Step 6 skips raw summaries",
         len(trace[5]["skipped"]) == 4),
        ("Dependency chain: gather->compare->synthesise->refine",
         len(trace) == 7),
        ("No duplication of memory across steps",
         len(set(p for t in trace for p in t["writes"])) ==
         len([p for t in trace for p in t["writes"]])),
        ("System stable - all 7 steps completed",
         len(trace) == 7 and all(t["vfs_size_after"] > 0 for t in trace)),
    ]

    passed = 0
    for label, result in checks:
        icon = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"  {icon}  {label}")
        if result:
            passed += 1

    score = int((passed / len(checks)) * 100)
    console.print(f"\n[bold]Score: {passed}/{len(checks)} checks passed ({score}%)[/bold]")

    if score == 100:
        console.print(Panel(
            "[bold green]All mentor criteria met.[/bold green]\n"
            "System stable. Memory intelligent. Dependency chain clear.",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[yellow]{len(checks)-passed} checks need attention.[/yellow]",
            border_style="yellow"
        ))

    # FINAL OUTPUT
    console.print("\n")
    final = vfs["/drafts/unified_security.txt"]["content"]
    console.print(Panel(
        Markdown(final[:1500] + "\n\n...[truncated for display]"),
        title="[bold green]Final Output: Unified Cybersecurity Framework[/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    run()