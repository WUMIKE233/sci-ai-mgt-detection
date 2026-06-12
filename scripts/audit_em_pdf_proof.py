"""Audit an Editorial Manager PDF proof before author approval.

The script is intentionally conservative: it blocks stale proofs that still
contain old submission metadata, old scope wording, or numbered-reference
traces known to have appeared in earlier generated proofs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import pypdf
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("pypdf is required: python -m pip install pypdf") from exc


FINAL_TITLE = "Calibrated Detection of AI-Generated Text Under Cross-Domain and Paraphrase-Style Distribution Shifts"
EXPECTED_SNIPPETS = [
    FINAL_TITLE,
    "MAGE paraphrase OOD",
    "Accuracy is 0.6518 on SemEval monolingual dev",
    "Dataset access instructions, preprocessing scripts, configuration files, and the locked aggregate-result summary",
    "eswa-submission-v1",
]
STALE_MARKERS = [
    "Adversarial Distribution Shifts",
    "Current preliminary evidence",
    "adversarial robustness",
    "Generated Preliminary",
    "Final formatted reference list should be generated",
    "Verified experiments are required",
]


def extract_text(pdf_path: Path) -> tuple[int, str]:
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return len(reader.pages), "\n".join(pages)


def check_pdf(pdf_path: Path) -> dict:
    if not pdf_path.exists():
        return {
            "overall_status": "BLOCKED",
            "pdf": str(pdf_path),
            "error": "PDF file does not exist.",
            "checks": [],
        }

    page_count, text = extract_text(pdf_path)
    checks = []

    def add(name: str, ok: bool, evidence: str, action: str) -> None:
        checks.append(
            {
                "name": name,
                "status": "pass" if ok else "blocker",
                "evidence": evidence,
                "action": action,
            }
        )

    add(
        "PDF text extracted",
        page_count > 0 and len(text.strip()) > 1000,
        f"pages={page_count}; text_chars={len(text)}",
        "Download the generated PDF proof again if text extraction fails.",
    )

    missing = [snippet for snippet in EXPECTED_SNIPPETS if snippet not in text]
    add(
        "Expected final-scope text is present",
        not missing,
        f"missing_snippets={missing}",
        "Rebuild the PDF proof after replacing manuscript files and updating Editorial Manager metadata.",
    )

    stale_hits = [marker for marker in STALE_MARKERS if marker.lower() in text.lower()]
    add(
        "No stale proof markers remain",
        not stale_hits,
        f"stale_hits={stale_hits}",
        "Do not approve this proof; update the submission files/metadata and rebuild the PDF.",
    )

    references_text = text.split("\nReferences", 1)[-1] if "\nReferences" in text else ""
    numeric_reference_hits = re.findall(r"\n\s*\d+\.\s+[A-Z][^\n]{10,}", references_text)
    add(
        "No obvious numbered-reference list remains",
        not numeric_reference_hits,
        f"numbered_reference_like_hits={numeric_reference_hits[:5]}",
        "Use the latest APA-style manuscript DOCX and rebuild the PDF.",
    )

    overall = "BLOCKED" if any(row["status"] == "blocker" for row in checks) else "PASS"
    return {
        "overall_status": overall,
        "pdf": str(pdf_path),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an Editorial Manager PDF proof.")
    parser.add_argument("pdf", type=Path, help="Path to the downloaded PDF proof.")
    parser.add_argument("--json-out", type=Path, default=Path("outputs/readiness/em_pdf_proof_audit.json"))
    args = parser.parse_args()

    result = check_pdf(args.pdf)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "overall_status": result["overall_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
