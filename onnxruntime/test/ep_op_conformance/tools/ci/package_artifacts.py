"""Package CI source/native fixtures using the reusable canonical-case archive tool."""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_cases_from_json import read_generated
from package_cases import create as package_cases
from package_cases import (
    extract_archive,
    inventory,
    seal,
    sha256,
    verify,
    verify_archive,
    verify_cases_archive,
    write_archive,
    write_json,
)

from ci.prepare_bundle import prepare

ROOT = Path(__file__).resolve().parents[2]


def check_generated(generated):
    metadata = read_generated(generated)
    if not re.fullmatch(r"[0-9a-f]{64}", metadata["input_digest"]):
        raise ValueError("Invalid generated case identity")
    return generated / "cases", metadata


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
        cases.mkdir()
        package_cases(
            positive,
            cases / "cases.zip",
            common["ort_version"],
            revision,
            source / "LICENSE",
            source / "ThirdPartyNotices.txt",
            provenance=generation,
        )
        scratch = output / "preparation"
        prepare(runner, runtime, scratch, positive, plugin)
        shutil.rmtree(scratch / "cases/positive")
        negative = scratch / "negative-fixtures"
        negative.mkdir()
        shutil.move(scratch / "cases", negative / "cases")
        shutil.move(scratch / "profiles", negative / "profiles")
        write_archive(negative, cases / "negative-fixtures.zip", "ci-fixtures", revision)
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
    canonical = verify_cases_archive(cases / "cases.zip", revision)
    verify_archive(cases / "negative-fixtures.zip", "ci-fixtures", revision)
    positive_files = {
        name.removeprefix("cases/"): digest for name, digest in canonical["files"].items() if name.startswith("cases/")
    }
    if (
        generation["input_digest"] != corpus["corpus_digest"]
        or positive_files != generation["files"]
        or canonical["provenance"] != generation
    ):
        raise ValueError("Case provenance mismatch")
    for name, digest in generation["inputs"].items():
        path = source / ("cases" if name == "pilot.json" else "tools") / name
        if sha256(path) != digest:
            raise ValueError(f"Case input differs from source kit: {name}")
    return revision


def stage(source, cases, fixture, runner, output, minimum_runtime=None, revision=None):
    revision = verify_all(source, cases, fixture, revision)
    runtime = minimum_runtime or fixture / "onnxruntime.dll"
    plugin = None if minimum_runtime else fixture / "example_plugin_ep.dll"
    with tempfile.TemporaryDirectory(prefix="ort-conformance-stage-") as temporary:
        root = Path(temporary)
        extract_archive(cases / "cases.zip", root / "canonical", "conformance-cases", revision)
        extract_archive(cases / "negative-fixtures.zip", root / "negative", "ci-fixtures", revision)
        prepare(runner, runtime, output, root / "canonical/cases", plugin)
        if plugin:
            actual = {
                name: digest for name, digest in inventory(output / "cases").items() if not name.startswith("positive/")
            }
            if actual != inventory(root / "negative/cases"):
                raise ValueError("Derived scenario bytes differ from the producer package")
            if inventory(output / "profiles") != inventory(root / "negative/profiles"):
                raise ValueError("Derived profiles differ from the producer package")
    shutil.copy2(cases / "generation.json", output / "generation.json")
    metadata = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    metadata["source_revision"] = revision
    metadata["package_manifests"] = {
        name: json.loads((path / "artifact.json").read_text(encoding="utf-8"))
        for name, path in (("source", source), ("cases", cases), ("fixture", fixture))
    }
    metadata["files"]["generation.json"] = sha256(output / "generation.json")
    write_json(output / "prepared.json", metadata)


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
        if arguments.command == "stage":
            stage(
                arguments.source,
                arguments.cases,
                arguments.fixture,
                arguments.runner,
                arguments.output,
                arguments.minimum_runtime,
                arguments.revision,
            )
        else:
            verify_all(arguments.source, arguments.cases, arguments.fixture, arguments.revision)
