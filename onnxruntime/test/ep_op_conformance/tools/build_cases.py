import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

from generate_cases import generate


def identity(definitions):
    sources = [
        definitions,
        Path(__file__),
        Path(__file__).with_name("generate_cases.py"),
    ]
    inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "onnx", "protobuf")}
    return {"inputs": inputs, "producer_packages": packages}


def materialize(definitions, output):
    provenance = identity(definitions)
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    destination = output / digest
    record = output / f"{digest}.json"
    if not destination.exists():
        staging = output / f"{digest}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        generate(definitions, staging)
        staging.rename(destination)
    elif not record.exists():
        raise ValueError("Incomplete generated case set; use a fresh output directory")
    files = {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    if record.exists() and json.loads(record.read_text(encoding="utf-8"))["files"] != files:
        raise ValueError("Generated case bytes changed; cases are immutable")
    metadata = {
        "schema_version": 1,
        "input_digest": digest,
        "source_revision": os.environ.get("GITHUB_SHA"),
        **provenance,
        "directory": digest,
        "files": files,
    }
    output.mkdir(parents=True, exist_ok=True)
    if not record.exists():
        record.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary = output / "current.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "current.json")


def verify_generated(definitions, output):
    metadata = json.loads((output / "current.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(identity(definitions), sort_keys=True).encode()).hexdigest()
    if metadata["input_digest"] != digest or metadata["directory"] != digest:
        raise ValueError("Generated case input identity mismatch")
    destination = output / digest
    files = {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    if files != metadata["files"]:
        raise ValueError("Generated case hashes differ; cases are immutable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--definitions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-key", action="store_true")
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.print_key:
        print(hashlib.sha256(json.dumps(identity(arguments.definitions), sort_keys=True).encode()).hexdigest())
    else:
        if arguments.output is None:
            parser.error("--output is required unless --print-key is set")
        if arguments.verify:
            verify_generated(arguments.definitions, arguments.output)
        else:
            materialize(arguments.definitions, arguments.output)
