"""Probe Hugging Face dataset repository metadata without downloading large files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe HF dataset metadata.")
    parser.add_argument("repo_id", nargs="+", help="Dataset repo id, e.g. yaful/MAGE")
    parser.add_argument("--output", type=Path, default=Path("data/dataset_probe.json"))
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    results = []
    for repo_id in args.repo_id:
        info = api.dataset_info(repo_id, files_metadata=True, timeout=args.timeout)
        card = info.cardData.to_dict() if hasattr(info.cardData, "to_dict") else info.cardData
        files = []
        for sibling in info.siblings:
            files.append(
                {
                    "path": sibling.rfilename,
                    "size": getattr(sibling, "size", None),
                    "lfs": getattr(sibling, "lfs", None),
                }
            )
        results.append(
            {
                "repo_id": repo_id,
                "sha": info.sha,
                "card_data": card,
                "files": files,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "repos": args.repo_id}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
