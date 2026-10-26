"""Package existing conformance cases without regenerating models or expected outputs."""

import argparse
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import onnx
from onnx import numpy_helper


def stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return stream_sha256(stream)


def safe_relative_path(relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Unsafe package path")
    path = PurePosixPath(relative)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != relative
        or ".." in path.parts
        or any(character in relative for character in '\\:<>"|?*')
        or any(ord(character) < 32 for character in relative)
        or any(part.endswith((".", " ")) for part in path.parts)
        or any(re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part) for part in path.parts)
    ):
        raise ValueError("Unsafe package path")
    return path


def inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Package root must be a directory, not a symlink: {root}")
    result = {}
    portable_names = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Package contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        safe_relative_path(relative)
        if relative.casefold() in portable_names:
            raise ValueError(f"Case-insensitive package path collision: {relative}")
        portable_names.add(relative.casefold())
        if path.is_file() and relative != "artifact.json":
            result[relative] = sha256(path)
    return result


def seal(root, kind, revision, **metadata):
    if not revision:
        raise ValueError("Package source revision is required")
    write_json(
        root / "artifact.json",
        {
            "schema_version": 1,
            "kind": kind,
            "source_revision": revision,
            **metadata,
            "files": inventory(root),
        },
    )


def validate_metadata(metadata, kind, revision=None):
    if metadata.get("schema_version") != 1 or metadata.get("kind") != kind:
        raise ValueError("Unsupported package schema or kind")
    if not metadata.get("source_revision") or (revision and metadata["source_revision"] != revision):
        raise ValueError("Package source revision mismatch")
    if not isinstance(metadata.get("files"), dict):
        raise ValueError("Package file inventory is required")
    for relative, digest in metadata["files"].items():
        safe_relative_path(relative)
        if relative == "artifact.json" or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid package file inventory")


def verify(root, kind, revision=None):
    metadata = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    validate_metadata(metadata, kind, revision)
    if inventory(root) != metadata["files"]:
        raise ValueError("Package file hashes or dependency closure mismatch")
    return metadata


def write_archive(root, output, kind, revision, **metadata):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive output must be outside its input directory")
    document = {"schema_version": 1, "kind": kind, "source_revision": revision, **metadata, "files": inventory(root)}
    validate_metadata(document, kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        try:
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for relative in [*document["files"], "artifact.json"]:
                    entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFREG | 0o644) << 16
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    with archive.open(entry, "w", force_zip64=True) as target:
                        if relative == "artifact.json":
                            target.write((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
                        else:
                            with (root / relative).open("rb") as source:
                                shutil.copyfileobj(source, target)
            destination.flush()
            verify_archive(output, kind, revision)
        except Exception:
            destination.close()
            output.unlink()
            raise


def read_archive(archive, kind, revision=None):
    names = set()
    for entry in archive.infolist():
        safe_relative_path(entry.filename)
        if entry.filename.casefold() in names or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("Duplicate or non-file archive entry")
        names.add(entry.filename.casefold())
    document = json.loads(archive.read("artifact.json"))
    validate_metadata(document, kind, revision)
    if {entry.filename for entry in archive.infolist()} != {*document["files"], "artifact.json"}:
        raise ValueError("Archive dependency closure mismatch")
    for relative, expected in document["files"].items():
        with archive.open(relative) as source:
            if stream_sha256(source) != expected:
                raise ValueError(f"Archive file hashes differ: {relative}")
    return document


def verify_archive(path, kind, revision=None):
    with zipfile.ZipFile(path) as archive:
        return read_archive(archive, kind, revision)


def extract_archive(path, output, kind, revision=None):
    if output.exists():
        raise ValueError("Output exists; packages are immutable")
    with zipfile.ZipFile(path) as archive:
        metadata = read_archive(archive, kind, revision)
        output.mkdir(parents=True)
        try:
            for entry in archive.infolist():
                target = output / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        except Exception:
            shutil.rmtree(output)
            raise
    return metadata


def validate_cases(root):
    if (root / "artifact.json").exists():
        raise ValueError("Unexpected artifact.json outside the case dependency closure")
    files = inventory(root)
    manifests = sorted(root.glob("*/case.json"))
    if not manifests:
        raise ValueError("No case manifests found")
    referenced = set()
    identities = set()
    descriptions = []

    def fields(value, required, optional=()):
        if not isinstance(value, dict) or not set(required).issubset(value) or value.keys() - {*required, *optional}:
            raise ValueError("Unsupported or missing case fields")

    for path in manifests:
        case = json.loads(path.read_text(encoding="utf-8"))
        fields(case, ("schema_version", "id", "model", "required_nodes", "datasets", "comparison", "source"))
        identity = case["id"]
        if case["schema_version"] != 1 or not isinstance(identity, str) or not identity:
            raise ValueError("Unsupported case version or empty ID")
        if identity.casefold() in identities:
            raise ValueError("Duplicate case ID")
        identities.add(identity.casefold())
        if not isinstance(case["source"], dict) or not case["source"]:
            raise ValueError("Case must retain source provenance")
        referenced.add(path.relative_to(root).as_posix())

        def case_path(relative, directory=path.parent):
            safe_relative_path(relative)
            target = directory / relative
            if not target.is_file():
                raise ValueError(f"Missing case dependency: {relative}")
            referenced.add(target.relative_to(root).as_posix())
            return target

        model = onnx.load_model(case_path(case["model"]), load_external_data=False)
        if model.functions or model.training_info or not model.graph.node or model.graph.sparse_initializer:
            raise ValueError("Only nonempty dense graphs without functions or training are supported")
        tensors = list(model.graph.initializer)
        nodes = set()
        for node in model.graph.node:
            if node.domain not in ("", "ai.onnx") or not node.name or node.name in nodes:
                raise ValueError("Expected uniquely named standard-domain nodes")
            nodes.add(node.name)
            for attribute in node.attribute:
                if attribute.type in (
                    onnx.AttributeProto.GRAPH,
                    onnx.AttributeProto.GRAPHS,
                    onnx.AttributeProto.SPARSE_TENSOR,
                    onnx.AttributeProto.SPARSE_TENSORS,
                ):
                    raise ValueError("Nested graphs and sparse attributes are not supported")
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                tensors.extend(attribute.tensors)
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
            if not value.type.HasField("tensor_type") or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                raise ValueError("Only float32 tensor model values are supported")
        required = case["required_nodes"]
        if not isinstance(required, list) or not required or any(not isinstance(node, str) for node in required):
            raise ValueError("Invalid required nodes")
        if len(set(required)) != len(required) or not set(required).issubset(nodes):
            raise ValueError("Required node absent or duplicated")
        if not isinstance(case["datasets"], list) or not case["datasets"]:
            raise ValueError("Case has no datasets")
        outputs = {value.name for value in model.graph.output}
        feeds = {value.name for value in model.graph.input} - {value.name for value in model.graph.initializer}
        fields(case["comparison"], outputs)
        for policy in case["comparison"].values():
            fields(policy, ("atol", "rtol", "nan_equal"))
            if type(policy["nan_equal"]) is not bool or any(
                type(policy[key]) not in (int, float) or not math.isfinite(policy[key]) or policy[key] < 0
                for key in ("atol", "rtol")
            ):
                raise ValueError("Invalid comparison policy")
        for dataset in case["datasets"]:
            fields(dataset, ("inputs", "outputs"))
            for kind, expected in (("inputs", feeds), ("outputs", outputs)):
                if not isinstance(dataset[kind], list):
                    raise ValueError("Dataset paths must be arrays")
                names = set()
                for relative in dataset[kind]:
                    tensor = onnx.load_tensor(case_path(relative))
                    if not tensor.name or tensor.name in names:
                        raise ValueError("Missing or duplicate tensor name")
                    names.add(tensor.name)
                    tensors.append(tensor)
                if names != expected or (kind == "outputs" and not names):
                    raise ValueError("Dataset names differ from model inputs/outputs")
        for tensor in tensors:
            if (
                tensor.data_type != onnx.TensorProto.FLOAT
                or tensor.external_data
                or tensor.data_location == onnx.TensorProto.EXTERNAL
                or tensor.HasField("segment")
                or tensor.int32_data
                or tensor.int64_data
                or tensor.uint64_data
                or tensor.double_data
                or tensor.string_data
            ):
                raise ValueError("Only inline dense float32 tensors are supported")
            count = math.prod(tensor.dims)
            if any(dimension < 0 for dimension in tensor.dims) or count > ((1 << 64) - 1) // 4:
                raise ValueError("Invalid tensor shape")
            if tensor.HasField("raw_data"):
                if tensor.float_data or len(tensor.raw_data) != count * 4:
                    raise ValueError("Invalid raw tensor data size")
            elif len(tensor.float_data) != count:
                raise ValueError("Invalid tensor element count")
            numpy_helper.to_array(tensor)
        descriptions.append(
            {
                "id": identity,
                "directory": path.parent.name,
                "ir_version": model.ir_version,
                "opsets": {opset.domain: opset.version for opset in model.opset_import},
            }
        )
    if referenced != set(files):
        raise ValueError("Unexpected files outside the case dependency closure")
    return descriptions


def create(cases, output, version, revision, license_path, notices, provenance=None):
    if not version or not revision:
        raise ValueError("Package version and source revision are required")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Producer provenance must be a JSON object")
    if output.resolve().is_relative_to(cases.resolve()):
        raise ValueError("Archive output must be outside its input directory")
    original = inventory(cases)
    descriptions = validate_cases(cases)
    with tempfile.TemporaryDirectory(prefix="ort-conformance-package-") as temporary:
        root = Path(temporary)
        shutil.copytree(cases, root / "cases")
        if inventory(root / "cases") != original:
            raise ValueError("Case files changed during packaging")
        shutil.copy2(license_path, root / "LICENSE")
        shutil.copy2(notices, root / "ThirdPartyNotices.txt")
        write_archive(
            root,
            output,
            "conformance-cases",
            revision,
            version=version,
            cases=descriptions,
            provenance=provenance or {},
        )


def verify_cases_archive(path, revision=None):
    with tempfile.TemporaryDirectory(prefix="ort-conformance-verify-") as temporary:
        root = Path(temporary) / "package"
        metadata = extract_archive(path, root, "conformance-cases", revision)
        if not metadata.get("version") or metadata.get("cases") != validate_cases(root / "cases"):
            raise ValueError("Case package metadata mismatch")
        expected = {"LICENSE", "ThirdPartyNotices.txt"}
        if {name for name in metadata["files"] if not name.startswith("cases/")} != expected:
            raise ValueError("Unexpected case package contents")
        return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser(
        "create", help="Validate and archive existing case directories; never run or regenerate."
    )
    for name in ("cases", "output", "license", "notices"):
        pack.add_argument(f"--{name}", required=True, type=Path)
    pack.add_argument("--version", required=True)
    pack.add_argument("--revision", required=True)
    pack.add_argument("--provenance", type=Path, help="Optional JSON producer provenance to retain in the manifest.")
    check = commands.add_parser("verify", help="Verify hashes, dependency closure and supported case structure.")
    check.add_argument("--archive", required=True, type=Path)
    check.add_argument("--revision")
    arguments = parser.parse_args()
    if arguments.command == "create":
        create(
            arguments.cases,
            arguments.output,
            arguments.version,
            arguments.revision,
            arguments.license,
            arguments.notices,
            json.loads(arguments.provenance.read_text()) if arguments.provenance else None,
        )
    else:
        verify_cases_archive(arguments.archive, arguments.revision)
