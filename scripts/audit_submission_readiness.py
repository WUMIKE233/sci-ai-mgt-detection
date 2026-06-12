"""Audit whether the SCI manuscript package is submission-ready.

The audit is intentionally conservative: preliminary evidence counts as
progress, but unresolved RESULT_REQUIRED markers, missing author declarations,
or incomplete benchmark processing keep the package blocked from submission.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


TEXT_EXTENSIONS = {".md", ".txt", ".json", ".bib", ".py", ".csv"}
PLACEHOLDER_PATTERNS = [
    r"RESULT_REQUIRED",
    r"\bTODO\b",
    r"\bTBD\b",
]
PLACEHOLDER_SCAN_EXCLUDES = {
    str(Path("docs/submission_docx_qa.md")),
    str(Path("docs/submission_readiness_report.md")),
}


def file_size(path: str | Path) -> int:
    target = Path(path)
    return target.stat().st_size if target.exists() and target.is_file() else 0


def exists_nonempty(path: str | Path) -> bool:
    target = Path(path)
    return target.exists() and target.is_file() and target.stat().st_size > 0


def count_bib_entries(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"@\w+\s*\{", text))


def count_words_markdown(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text))


def scan_placeholders(paths: list[Path]) -> tuple[list[dict], Counter]:
    rows = []
    counts = Counter()
    patterns = [re.compile(pattern) for pattern in PLACEHOLDER_PATTERNS]
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            if str(path) in PLACEHOLDER_SCAN_EXCLUDES:
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                for pattern in patterns:
                    if pattern.search(line):
                        rel = str(path)
                        rows.append({"file": rel, "line": line_no, "marker": pattern.pattern, "text": line.strip()[:220]})
                        counts[rel] += 1
    return rows, counts


def check(name: str, category: str, status: str, evidence: str, action: str = "") -> dict:
    return {
        "category": category,
        "name": name,
        "status": status,
        "evidence": evidence,
        "action": action,
    }


def status_from_bool(ok: bool, blocker: bool = True) -> str:
    if ok:
        return "pass"
    return "blocker" if blocker else "warn"


def build_checks(placeholder_rows: list[dict], placeholder_counts: Counter) -> list[dict]:
    checks = []

    manuscript = Path("manuscript/manuscript_draft.md")
    declarations = Path("manuscript/declarations.md")
    cover = Path("manuscript/cover_letter_draft.md")
    highlights = Path("manuscript/highlights.md")
    author_metadata = Path("docs/author_metadata_template.md")
    references = Path("references/seed_references.bib")
    word_count = count_words_markdown(manuscript)
    bib_count = count_bib_entries(references)

    checks.append(
        check(
            "Manuscript draft exists",
            "manuscript",
            status_from_bool(exists_nonempty(manuscript)),
            f"{manuscript} size={file_size(manuscript)} bytes; word_count≈{word_count}",
            "Maintain the draft as the primary manuscript file.",
        )
    )
    checks.append(
        check(
            "Manuscript RESULT_REQUIRED markers resolved",
            "manuscript",
            status_from_bool(placeholder_counts.get(str(manuscript), 0) == 0),
            f"{placeholder_counts.get(str(manuscript), 0)} markers in {manuscript}",
            "Replace each marker only after the corresponding evidence is locked.",
        )
    )
    checks.append(
        check(
            "Declarations resolved",
            "submission package",
            status_from_bool(exists_nonempty(declarations) and placeholder_counts.get(str(declarations), 0) == 0),
            f"{declarations} markers={placeholder_counts.get(str(declarations), 0)}",
            "Fill funding, competing interests, data availability, code availability, and CRediT roles.",
        )
    )
    checks.append(
        check(
            "Cover letter resolved",
            "submission package",
            status_from_bool(exists_nonempty(cover) and placeholder_counts.get(str(cover), 0) == 0),
            f"{cover} markers={placeholder_counts.get(str(cover), 0)}",
            "Insert target journal, author list, affiliation, word count, figure/table count, and corresponding author.",
        )
    )
    checks.append(
        check(
            "Highlights prepared",
            "submission package",
            status_from_bool(exists_nonempty(highlights)),
            f"{highlights} size={file_size(highlights)} bytes",
            "Revise highlights after final claims are locked.",
        )
    )
    docx_package_files = [
        "submission_package/manuscript_draft_eswa.docx",
        "submission_package/cover_letter_draft.docx",
        "submission_package/declarations_draft.docx",
        "submission_package/author_metadata_template.docx",
        "submission_package/submission_docx_manifest.json",
    ]
    missing_docx_package_files = [path for path in docx_package_files if not exists_nonempty(path)]
    checks.append(
        check(
            "DOCX submission drafts generated",
            "submission package",
            status_from_bool(not missing_docx_package_files, blocker=False),
            "missing=" + (", ".join(missing_docx_package_files) if missing_docx_package_files else "none"),
            "Regenerate with the bundled Python runtime and scripts/build_submission_docx.py after final manuscript edits.",
        )
    )
    checks.append(
        check(
            "Reference list sufficient for full paper",
            "literature",
            status_from_bool(bib_count >= 15),
            f"{references} entries={bib_count}",
            "Add verified DOI-checked calibration, conformal prediction, detector, and recent journal references.",
        )
    )

    required_assets = [
        "manuscript/tables/table1_preliminary_metrics_with_ci.md",
        "manuscript/tables/table2_worst_subgroups.md",
        "manuscript/figures/figure1_accuracy_with_ci.png",
        "manuscript/figures/figure2_ece_with_ci.png",
        "manuscript/figures/figure3_worst_subgroups.png",
        "manuscript/figures/figure_captions.md",
    ]
    missing_assets = [path for path in required_assets if not exists_nonempty(path)]
    checks.append(
        check(
            "Current manuscript assets generated",
            "figures and tables",
            status_from_bool(not missing_assets, blocker=False),
            "missing=" + (", ".join(missing_assets) if missing_assets else "none"),
            "Regenerate with python scripts/make_manuscript_assets.py after final predictions are locked.",
        )
    )

    evidence_files = [
        "docs/preliminary_experiment_summary.md",
        "docs/bootstrap_confidence_intervals.md",
        "docs/error_analysis.md",
        "docs/semeval_preliminary_report.md",
        "docs/semeval_seed_sweep_report.md",
        "docs/neural_seed_sweep_report.md",
        "docs/environment_snapshot.md",
        "docs/submission_package_checklist.md",
        "outputs/experiment_summary/preliminary_metrics.csv",
        "outputs/statistical_analysis/bootstrap_confidence_intervals.csv",
        "outputs/error_analysis/subgroup_error_summary.csv",
        "outputs/semeval_seed_sweep/seed_sweep_summary.csv",
        "outputs/neural_seed_sweep/neural_seed_sweep_summary.csv",
        "outputs/reproducibility/environment_snapshot.json",
    ]
    missing_evidence = [path for path in evidence_files if not exists_nonempty(path)]
    checks.append(
        check(
            "Current preliminary evidence files present",
            "evidence",
            status_from_bool(not missing_evidence, blocker=False),
            "missing=" + (", ".join(missing_evidence) if missing_evidence else "none"),
            "Keep evidence files synchronized with regenerated predictions.",
        )
    )

    semeval_mono = [
        "data/raw/semeval2024_task8/subtaskA/SubtaskA/subtaskA_train_monolingual.jsonl",
        "data/raw/semeval2024_task8/subtaskA/SubtaskA/subtaskA_dev_monolingual.jsonl",
    ]
    checks.append(
        check(
            "SemEval monolingual source files present",
            "data",
            status_from_bool(all(exists_nonempty(path) for path in semeval_mono), blocker=False),
            "; ".join(f"{path} size={file_size(path)}" for path in semeval_mono),
            "Use these files for current SemEval monolingual pre-experiments.",
        )
    )
    multilingual_train = Path("data/raw/semeval2024_task8/subtaskA/SubtaskA/subtaskA_train_multilingual.jsonl")
    partial_multilingual = list(Path("data/raw/semeval2024_task8/subtaskA/SubtaskA").glob("subtaskA_train_multilingual*.part"))
    checks.append(
        check(
            "SemEval multilingual train complete",
            "data",
            status_from_bool(exists_nonempty(multilingual_train)),
            f"complete={multilingual_train.exists()}; partial_files={[str(p) for p in partial_multilingual]}",
            "Resume or redownload multilingual train before claiming full SemEval/M4 coverage.",
        )
    )

    mage_full = ["data/raw/mage/train.csv", "data/raw/mage/valid.csv", "data/raw/mage/test.csv"]
    checks.append(
        check(
            "MAGE full train/valid/test present",
            "data",
            status_from_bool(all(exists_nonempty(path) for path in mage_full)),
            "; ".join(f"{path} exists={Path(path).exists()}" for path in mage_full),
            "Download/verify full MAGE files or predeclare a justified sampled protocol.",
        )
    )

    neural_seed_report = Path("docs/neural_seed_sweep_report.md")
    checks.append(
        check(
            "Repeated-seed neural baselines complete",
            "statistics",
            status_from_bool(exists_nonempty(neural_seed_report)),
            f"{neural_seed_report} exists={neural_seed_report.exists()}",
            "Run repeated-seed DistilBERT or stronger encoder experiments after selecting final model scope.",
        )
    )

    public_code_record = Path("docs/public_code_availability_record.md")
    public_code_text = public_code_record.read_text(encoding="utf-8", errors="replace").lower() if public_code_record.exists() else ""
    public_code_ready = (
        exists_nonempty(public_code_record)
        and "current status: not assigned" not in public_code_text
        and (
            "https://" in public_code_text
            or "http://" in public_code_text
            or "doi" in public_code_text
            or "journal-compliant exception" in public_code_text
        )
    )
    local_code_archive = Path("outputs/reproducibility/sci_ai_mgt_detection_code_package.zip")
    local_code_manifest = Path("outputs/reproducibility/code_archive_manifest.json")
    checks.append(
        check(
            "Local code archive package prepared",
            "reproducibility",
            status_from_bool(exists_nonempty(local_code_archive) and exists_nonempty(local_code_manifest), blocker=False),
            f"{local_code_archive} exists={local_code_archive.exists()}; {local_code_manifest} exists={local_code_manifest.exists()}",
            "Publish or archive this package before final submission.",
        )
    )
    checks.append(
        check(
            "Public code/archive location ready",
            "reproducibility",
            status_from_bool(public_code_ready),
            f"{public_code_record} exists={public_code_record.exists()}; public_url_or_doi_or_exception={public_code_ready}",
            "Create a public repository, archive DOI, or explicit journal-compliant code-availability exception before submission.",
        )
    )
    journal_artifacts = sorted(Path("docs").glob("journal_verification_*.md"))
    journal_evidence = (
        f"public verification artifacts={[str(path) for path in journal_artifacts]}; final Web of Science JCR/CAS screenshot or export still recommended"
        if journal_artifacts
        else "No current journal verification artifact found."
    )
    checks.append(
        check(
            "Target journal current verification complete",
            "journal",
            status_from_bool(False),
            journal_evidence,
            "Verify the selected SCI Q1/一区 journal in Web of Science JCR or CAS partition records close to submission.",
        )
    )
    author_metadata_text = author_metadata.read_text(encoding="utf-8", errors="replace").lower() if author_metadata.exists() else ""
    author_metadata_ready = (
        exists_nonempty(author_metadata)
        and "wuzhuoxian" in author_metadata_text
        and "guangdong neusoft university" in author_metadata_text
        and "funding source: none" in author_metadata_text
        and "competing interests" in author_metadata_text
        and "credit roles" in author_metadata_text
        and placeholder_counts.get(str(declarations), 0) == 0
        and placeholder_counts.get(str(cover), 0) == 0
    )
    checks.append(
        check(
            "Author/funding/ethics metadata complete",
            "submission package",
            status_from_bool(author_metadata_ready),
            f"{author_metadata} exists={author_metadata.exists()}; author_metadata_ready={author_metadata_ready}",
            "Before final upload, enter the corresponding author's email and postal address in the journal submission system.",
        )
    )
    checks.append(
        check(
            "No unresolved placeholders anywhere in tracked writing files",
            "global",
            status_from_bool(len(placeholder_rows) == 0),
            f"placeholder_rows={len(placeholder_rows)}",
            "Use the placeholder table below as the task list.",
        )
    )
    return checks


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, checks: list[dict], placeholders: list[dict]) -> None:
    counts = Counter(row["status"] for row in checks)
    overall = "BLOCKED" if counts["blocker"] else ("WARN" if counts["warn"] else "PASS")
    lines = [
        "# Submission Readiness Audit",
        "",
        f"Overall status: **{overall}**",
        "",
        "This audit is conservative. Preliminary experiments count as progress, but unresolved placeholders, incomplete benchmark processing, missing author metadata, and missing journal verification block submission.",
        "",
        "## Gate Summary",
        "",
        f"- Blockers: {counts['blocker']}",
        f"- Warnings: {counts['warn']}",
        f"- Passes: {counts['pass']}",
        "",
        "## Checks",
        "",
        "| Category | Check | Status | Evidence | Action |",
        "|---|---|---|---|---|",
    ]
    for row in checks:
        lines.append(
            "| {category} | {name} | {status} | {evidence} | {action} |".format(
                category=row["category"],
                name=row["name"],
                status=row["status"],
                evidence=str(row["evidence"]).replace("|", "\\|"),
                action=str(row["action"]).replace("|", "\\|"),
            )
        )

    lines.extend(["", "## Highest-Priority Next Actions", ""])
    for row in [item for item in checks if item["status"] == "blocker"][:8]:
        lines.append(f"- **{row['name']}**: {row['action']}")

    lines.extend(
        [
            "",
            "## Placeholder Inventory",
            "",
            f"Total placeholder rows: `{len(placeholders)}`",
            "",
            "| File | Line | Marker | Text |",
            "|---|---:|---|---|",
        ]
    )
    for row in placeholders[:120]:
        lines.append(
            "| {file} | {line} | {marker} | {text} |".format(
                file=row["file"].replace("|", "\\|"),
                line=row["line"],
                marker=row["marker"].replace("|", "\\|"),
                text=row["text"].replace("|", "\\|"),
            )
        )
    if len(placeholders) > 120:
        lines.append(f"| ... | ... | ... | {len(placeholders) - 120} more placeholder rows omitted from markdown preview. |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scan_roots = [
        Path("manuscript"),
        Path("docs"),
        Path("README.md"),
        Path("data/DATA_SOURCES.md"),
        Path("scripts/experiment_commands.md"),
    ]
    placeholders, placeholder_counts = scan_placeholders(scan_roots)
    checks = build_checks(placeholders, placeholder_counts)

    output_dir = Path("outputs/readiness")
    output_dir.mkdir(parents=True, exist_ok=True)
    checks_csv = output_dir / "submission_readiness_checks.csv"
    placeholders_csv = output_dir / "submission_placeholders.csv"
    json_path = output_dir / "submission_readiness.json"
    report_path = Path("docs/submission_readiness_report.md")

    write_csv(checks_csv, checks, ["category", "name", "status", "evidence", "action"])
    write_csv(placeholders_csv, placeholders, ["file", "line", "marker", "text"])
    status_counts = Counter(row["status"] for row in checks)
    payload = {
        "overall_status": "BLOCKED" if status_counts["blocker"] else ("WARN" if status_counts["warn"] else "PASS"),
        "status_counts": dict(status_counts),
        "checks": checks,
        "placeholder_count": len(placeholders),
        "outputs": {
            "checks_csv": str(checks_csv),
            "placeholders_csv": str(placeholders_csv),
            "report": str(report_path),
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report_path, checks, placeholders)
    print(json.dumps({"json": str(json_path), "report": str(report_path), "overall_status": payload["overall_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
