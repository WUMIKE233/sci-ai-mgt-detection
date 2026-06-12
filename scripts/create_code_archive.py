"""Create a submission-oriented code archive without raw datasets."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


INCLUDE_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "operation_log.md",
]
INCLUDE_DIRS = [
    "configs",
    "docs",
    "manuscript",
    "references",
    "scripts",
]
INCLUDE_DATA_FILES = [
    "data/DATA_SOURCES.md",
    "data/processed/mage/metadata.json",
    "data/processed/raid/metadata.json",
    "data/processed/semeval2024_task8/semeval_subtaskA_monolingual_metadata.json",
]
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
}
EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    return True


def collect_files() -> list[Path]:
    files: list[Path] = []
    for file_name in INCLUDE_FILES:
        path = Path(file_name)
        if path.exists() and path.is_file():
            files.append(path)
    for dir_name in INCLUDE_DIRS:
        root = Path(dir_name)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and should_include(path):
                files.append(path)
    for file_name in INCLUDE_DATA_FILES:
        path = Path(file_name)
        if path.exists() and path.is_file():
            files.append(path)
    return sorted(set(files), key=lambda item: item.as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(path: Path, archive_name: str, manifest_name: str) -> None:
    lines = [
        "# Code Archive README",
        "",
        "This folder contains a local submission-oriented code archive for the SCI manuscript project.",
        "",
        "## Files",
        "",
        f"- `{archive_name}`: ZIP archive containing scripts, configs, manuscript drafts, documentation, references, and small metadata files.",
        f"- `{manifest_name}`: machine-readable manifest with file sizes and SHA-256 checksums.",
        "",
        "## Exclusions",
        "",
        "- Raw datasets under `data/raw/` are excluded.",
        "- Processed benchmark extracts under `data/processed/` are excluded except for small metadata files.",
        "- Generated outputs under `outputs/` are excluded except for this archive folder.",
        "- Model checkpoints, cache folders, local Word submission files, private operation logs, and Python bytecode are excluded from the public repository.",
        "",
        "## Submission Boundary",
        "",
        "Public code repository: https://github.com/WUMIKE233/sci-ai-mgt-detection",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs/reproducibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "sci_ai_mgt_detection_code_package.zip"
    manifest_path = output_dir / "code_archive_manifest.json"
    readme_path = output_dir / "README.md"

    files = collect_files()
    manifest_files = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.as_posix())
            manifest_files.append(
                {
                    "path": path.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path),
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "file_count": len(manifest_files),
        "files": manifest_files,
        "excluded": ["data/raw/", "data/processed/", "outputs/", "models/", "checkpoints/", "__pycache__/", "submission_package/"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(readme_path, archive_path.name, manifest_path.name)
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "manifest": str(manifest_path),
                "readme": str(readme_path),
                "file_count": len(manifest_files),
                "archive_size": archive_path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
