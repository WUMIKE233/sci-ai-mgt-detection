"""Download the core MAGE CSV files from Hugging Face.

The script downloads train.csv, valid.csv, and test.csv into data/raw/mage by
default. It keeps dataset acquisition reproducible and avoids committing large
raw files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_FILES = ["train.csv", "valid.csv", "test.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download core MAGE CSV files.")
    parser.add_argument("--repo-id", default="yaful/MAGE")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/mage"))
    parser.add_argument("--files", default=",".join(DEFAULT_FILES))
    return parser.parse_args()


def parse_files(value: str) -> list[str]:
    files = [item.strip() for item in value.split(",") if item.strip()]
    return files or DEFAULT_FILES


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in parse_files(args.files):
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=filename,
            local_dir=args.output_dir,
        )
        target = Path(path)
        downloaded.append(
            {
                "file": filename,
                "path": str(target),
                "size": target.stat().st_size if target.exists() else None,
            }
        )
    print(json.dumps({"output_dir": str(args.output_dir), "downloaded": downloaded}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
