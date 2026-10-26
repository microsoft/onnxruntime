import argparse
import json
import re
import shutil
from pathlib import Path, PurePosixPath

from prepare_ci import prepare, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Package contains a symlink: {path}")
        if path.is_file() and path != root / "artifact.json":
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def seal(root, kind, revision, **metadata):
    write_json(
        root / "artifact.json",
        {"schema_version": 1, "kind": kind, "source_revision": revision, **metadata, "files": inventory(root)},
    )


def verify(root, kind, revision=None):
    metadata = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["kind"] != kind:
        raise ValueError("Unsupported package schema or kind")
    if not metadata["source_revision"] or (revision and metadata["source_revision"] != revision):
        raise ValueError("Package source revision mismatch")
    for relative in metadata["files"]:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
            raise ValueError("Unsafe package path")
    if inventory(root) != metadata["files"]:
        raise ValueError("Package file hashes or dependency closure mismatch")
    return metadata


def check_generated(generated):
    metadata = json.loads((generated / "current.json").read_text(encoding="utf-8"))
    directory = metadata["directory"]
    if not re.fullmatch(r"[0-9a-f]{64}", directory) or directory != metadata["input_digest"]:
        raise ValueError("Invalid generated case identity")
    positive = generated / directory
    if inventory(positive) != metadata["files"]:
        raise ValueError("Generated case hash mismatch")
    return positive, metadata


def create(generated, runtime, plugin, output, revision, runner):
    if output.exists():
        raise ValueError("Output exists; packages are immutable")
    positive, generation = check_generated(generated)
    if generation["source_revision"] not in (None, revision):
        raise ValueError("Generated case source revision mismatch")
    source = ROOT.parents[2]
    try:
        kit = output / "source-kit"
        kit.mkdir(parents=True)
        shutil.copy2(ROOT / "CMakeLists.txt", kit)
        shutil.copy2(ROOT / "SOURCE_KIT.md", kit / "README.md")
        for directory in ("runner", "tools", "examples", "cases"):
            shutil.copytree(ROOT / directory, kit / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        headers = source / "include/onnxruntime/core/session"
        (kit / "include").mkdir()
        for header in headers.glob("*.h"):
            shutil.copy2(header, kit / "include" / header.name)
        for name in ("LICENSE", "ThirdPartyNotices.txt", "VERSION_NUMBER"):
            shutil.copy2(source / name, kit / name)
        shutil.copy2(
            source
            / "docs/design/webgpu_ep_extraction/test_ownership_and_conformance/ep_operator_conformance_design.md",
            kit / "CONTRACT.md",
        )
        api_header = (headers / "onnxruntime_c_api.h").read_text(encoding="utf-8")
        source_api = int(re.search(r"#define ORT_API_VERSION (\d+)", api_header)[1])
        common = {
            "ort_version": (source / "VERSION_NUMBER").read_text().strip(),
            "minimum_runtime_api": 24,
            "source_header_api": source_api,
        }
        seal(
            kit, "source-kit", revision, **common, dependency_build="CMake FetchContent hash-pinned public dependencies"
        )
        cases = output / "cases"
        scratch = output / "preparation"
        prepare(runner, runtime, scratch, positive, plugin)
        shutil.move(scratch / "cases", cases)
        shutil.move(scratch / "profiles", cases / "profiles")
        shutil.rmtree(scratch)
        write_json(cases / "generation.json", generation)
        seal(
            cases,
            "cases",
            revision,
            **common,
            corpus_digest=generation["input_digest"],
            producer_packages=generation["producer_packages"],
        )
        fixture = output / "fixture"
        fixture.mkdir()
        if runtime.name != "onnxruntime.dll" or plugin.name != "example_plugin_ep.dll":
            raise ValueError("Expected matching Windows x64 runtime and example_plugin_ep fixture")
        for binary in (runtime, plugin, runtime.with_name("onnxruntime_providers_shared.dll")):
            if binary in (runtime, plugin) or binary.exists():
                shutil.copy2(binary, fixture / binary.name)
        seal(
            fixture,
            "fixture",
            revision,
            **common,
            host_prerequisite="Windows x64; MSVC 2015-2022 redistributable",
            plugin_runtime_contract="Runtime and example_plugin_ep built together in this source job",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def verify_all(source, cases, fixture, revision=None):
    kit = verify(source, "source-kit", revision)
    revision = kit["source_revision"]
    corpus = verify(cases, "cases", revision)
    native = verify(fixture, "fixture", revision)
    for field in ("ort_version", "minimum_runtime_api", "source_header_api"):
        if kit[field] != corpus[field] or kit[field] != native[field]:
            raise ValueError(f"Package compatibility metadata mismatch: {field}")
    generation = json.loads((cases / "generation.json").read_text(encoding="utf-8"))
    if generation["input_digest"] != corpus["corpus_digest"] or inventory(cases / "positive") != generation["files"]:
        raise ValueError("Case provenance mismatch")
    for name, digest in generation["inputs"].items():
        path = source / ("cases" if name == "pilot.json" else "tools") / name
        if sha256(path) != digest:
            raise ValueError(f"Case input differs from source kit: {name}")
    return revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package or verify narrow EP op conformance CI artifacts")
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("create")
    for name in ("generated", "runtime", "plugin", "output"):
        pack.add_argument(f"--{name}", required=True, type=Path)
    pack.add_argument("--runner", required=True, type=Path)
    pack.add_argument("--revision", required=True)
    for command in ("verify", "stage"):
        command_parser = commands.add_parser(command)
        for name in ("source", "cases", "fixture"):
            command_parser.add_argument(f"--{name}", required=True, type=Path)
        command_parser.add_argument("--revision")
        if command == "stage":
            command_parser.add_argument("--runner", required=True, type=Path)
            command_parser.add_argument("--output", required=True, type=Path)
            command_parser.add_argument("--minimum-runtime", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "create":
        create(
            arguments.generated,
            arguments.runtime,
            arguments.plugin,
            arguments.output,
            arguments.revision,
            arguments.runner,
        )
    else:
        revision = verify_all(arguments.source, arguments.cases, arguments.fixture, arguments.revision)
        if arguments.command == "stage":
            runtime = arguments.minimum_runtime or arguments.fixture / "onnxruntime.dll"
            plugin = None if arguments.minimum_runtime else arguments.fixture / "example_plugin_ep.dll"
            prepare(arguments.runner, runtime, arguments.output, arguments.cases / "positive", plugin)
            if plugin:
                expected_cases = {
                    name: digest
                    for name, digest in inventory(arguments.cases).items()
                    if name not in ("generation.json",) and not name.startswith("profiles/")
                }
                if inventory(arguments.output / "cases") != expected_cases:
                    raise ValueError("Derived scenario bytes differ from the producer package")
                if inventory(arguments.output / "profiles") != inventory(arguments.cases / "profiles"):
                    raise ValueError("Derived profiles differ from the producer package")
            shutil.copy2(arguments.cases / "generation.json", arguments.output / "generation.json")
            metadata = json.loads((arguments.output / "prepared.json").read_text(encoding="utf-8"))
            metadata["source_revision"] = revision
            metadata["package_manifests"] = {
                name: json.loads((path / "artifact.json").read_text(encoding="utf-8"))
                for name, path in (
                    ("source", arguments.source),
                    ("cases", arguments.cases),
                    ("fixture", arguments.fixture),
                )
            }
            metadata["files"]["generation.json"] = sha256(arguments.output / "generation.json")
            write_json(arguments.output / "prepared.json", metadata)
