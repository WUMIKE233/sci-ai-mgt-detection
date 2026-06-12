"""Capture a lightweight reproducibility environment snapshot."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


KEY_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "gdown",
]


def run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return completed.returncode, completed.stdout.strip() or completed.stderr.strip()


def package_versions() -> dict[str, str]:
    code, output = run([sys.executable, "-m", "pip", "freeze"])
    versions = {}
    if code != 0:
        return {"pip_freeze_error": output}
    by_name = {}
    for line in output.splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        by_name[name.lower().replace("_", "-")] = version
    for package in KEY_PACKAGES:
        versions[package] = by_name.get(package.lower(), "not installed")
    return versions


def torch_info() -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment capture only
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    info = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        info["device_count"] = int(torch.cuda.device_count())
        info["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return info


def write_markdown(snapshot: dict, path: Path) -> None:
    lines = [
        "# Reproducibility Environment Snapshot",
        "",
        "This file records the local environment used for the current preliminary experiments. Final submission should regenerate it after the experiment protocol is locked.",
        "",
        "## Runtime",
        "",
        f"- Python executable: `{snapshot['python']['executable']}`",
        f"- Python version: `{snapshot['python']['version']}`",
        f"- Platform: `{snapshot['platform']['platform']}`",
        f"- Processor: `{snapshot['platform']['processor']}`",
        "",
        "## PyTorch / CUDA",
        "",
    ]
    torch = snapshot["torch"]
    if torch.get("available"):
        lines.extend(
            [
                f"- PyTorch: `{torch.get('version')}`",
                f"- CUDA available: `{torch.get('cuda_available')}`",
                f"- CUDA version: `{torch.get('cuda_version')}`",
                f"- CUDA devices: `{', '.join(torch.get('devices', [])) or 'none'}`",
            ]
        )
    else:
        lines.append(f"- PyTorch unavailable: `{torch.get('error')}`")

    lines.extend(["", "## Key Package Versions", "", "| Package | Version |", "|---|---|"])
    for name, version in snapshot["packages"].items():
        lines.append(f"| {name} | {version} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This environment snapshot supports reproducibility for the current preliminary package.",
            "- It is not a locked artifact for final submission until the final data processing and model scope are frozen.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs/reproducibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "platform": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "torch": torch_info(),
        "packages": package_versions(),
    }
    json_path = output_dir / "environment_snapshot.json"
    md_path = Path("docs/environment_snapshot.md")
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(snapshot, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
