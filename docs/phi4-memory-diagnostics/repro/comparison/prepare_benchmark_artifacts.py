# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Download dependencies must not be imported during host-only plans.
# ruff: noqa: PLC0415
"""Host-only planning and explicit, optional artifact download; never run inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

ORT_REPOSITORY = "microsoft/Phi-4-mini-instruct-onnx"
GGUF_REPOSITORY = "unsloth/Phi-4-mini-instruct-GGUF"
ORT_VARIANT = "gpu/gpu-int4-rtn-block-32"
GGUF_FILENAME = "Phi-4-mini-instruct-Q4_K_M.gguf"
ORT_REVISION = "fc04c8f93df696602fd9f300a30d1bf2e3081347"
GGUF_REVISION = "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80"


def revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("revision must be a full 40-character commit SHA")
    return value.lower()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="No downloads, optional imports, or writes")
    action.add_argument("--download", action="store_true", help="Explicitly opt in to artifact downloads")
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ort-revision", type=revision, default=ORT_REVISION)
    parser.add_argument("--gguf-revision", type=revision, default=GGUF_REVISION)
    args = parser.parse_args(argv)
    args.download_root = args.download_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.download_root, args.output_dir):
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            parser.error(f"Not a directory: {ancestor}")
    if (args.output_dir / "artifact-manifest.json").exists():
        parser.error("artifact-manifest.json already exists; choose a fresh --output-dir")
    return args


def download_plan(args: argparse.Namespace) -> list[dict]:
    return [
        {
            "repo_id": ORT_REPOSITORY,
            "revision": args.ort_revision,
            "allow_patterns": [f"{ORT_VARIANT}/*"],
            "local_dir": str(args.download_root / "ort"),
        },
        {
            "repo_id": GGUF_REPOSITORY,
            "revision": args.gguf_revision,
            "allow_patterns": [GGUF_FILENAME],
            "local_dir": str(args.download_root / "gguf"),
        },
    ]


def file_record(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = download_plan(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0

    from huggingface_hub import snapshot_download
    from run_ort_llama_benchmark import require_file, validate_ort_model

    for entry in plan:
        snapshot_download(**entry)
    ort_path = args.download_root / "ort" / ORT_VARIANT
    gguf_path = args.download_root / "gguf" / GGUF_FILENAME
    validate_ort_model(ort_path)
    require_file(gguf_path)
    manifest = {
        "schema_version": 2,
        "model": {
            "id": "microsoft/Phi-4-mini-instruct",
            "ort_repository": ORT_REPOSITORY,
            "ort_revision": args.ort_revision,
            "ort_variant": ORT_VARIANT,
            "ort_directory": str(ort_path),
            "ort_config": json.loads((ort_path / "genai_config.json").read_text(encoding="utf-8")),
            "ort_files": [file_record(path, ort_path) for path in sorted(ort_path.iterdir()) if path.is_file()],
            "gguf_repository": GGUF_REPOSITORY,
            "gguf_revision": args.gguf_revision,
            "gguf": {"file": str(gguf_path), **file_record(gguf_path, gguf_path.parent)},
        },
        "comparison_requirements": [
            "ORT INT4 RTN block 32 and GGUF Q4_K_M are not identical quantizers.",
            "Keep this manifest with benchmark outputs; local modifications change artifact identity.",
        ],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "huggingface_hub": version("huggingface_hub"),
            "historical_huggingface_hub_version": "unknown",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "artifact-manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ORT model: {ort_path}\nGGUF model: {gguf_path}\nManifest: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
