"""Create portable operator cases from authored JSON, optionally using a build cache."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def validate_fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"Expected a definition object, got {type(value).__name__}")
    required_fields = set(required)
    missing = required_fields - value.keys()
    unsupported = value.keys() - required_fields - set(optional)
    details = []
    if missing:
        details.append(f"missing: {', '.join(sorted(missing))}")
    if unsupported:
        details.append(f"unsupported: {', '.join(sorted(unsupported))}")
    if details:
        raise ValueError(f"Invalid definition fields: {'; '.join(details)}")


def case_directory_name(identity):
    if not isinstance(identity, str) or not re.fullmatch(
        r"(?:[A-Za-z0-9_-]+\.)+[A-Za-z0-9_]+\.opset[1-9][0-9]*\.[A-Za-z0-9_]+", identity
    ):
        raise ValueError(
            "Invalid case ID: expected <domain>.<operator>.opset<version>.<variant> using portable ASCII "
            "components and an alphanumeric/underscore variant"
        )
    return f"case-{identity}"


def tensor(definition, allow_initializer=False):
    validate_fields(definition, ("name", "shape", "values"), ("initializer",) if allow_initializer else ())
    shape = definition["shape"]
    if (
        not isinstance(definition["name"], str)
        or not definition["name"]
        or not isinstance(shape, list)
        or any(type(dim) is not int or dim < 0 for dim in shape)
    ):
        raise ValueError("Invalid tensor name or shape")
    if "initializer" in definition and type(definition["initializer"]) is not bool:
        raise ValueError("Initializer flag must be boolean")
    if not isinstance(definition["values"], list) or any(
        type(value) not in (int, float) for value in definition["values"]
    ):
        raise ValueError("Tensor values must be a flat numeric array")
    values = np.asarray(definition["values"], dtype=np.float32)
    if values.size != math.prod(shape) or not np.isfinite(values).all():
        raise ValueError("Invalid float32 tensor values")
    return numpy_helper.from_array(values.reshape(shape), name=definition["name"])


def value_info(value):
    return helper.make_tensor_value_info(value.name, value.data_type, list(value.dims))


def generate(definitions: Path, destination: Path):
    if destination.exists():
        raise ValueError("Destination already exists; generated cases are immutable")
    document = json.loads(definitions.read_text(encoding="utf-8"))
    validate_fields(document, ("schema_version", "cases"))
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["cases"], list)
        or not document["cases"]
    ):
        raise ValueError("Unsupported definition version or empty case set")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ep-op-conformance-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "cases"
        staging.mkdir()
        write_cases(document["cases"], definitions, staging)
        staging.rename(destination)


def write_cases(cases, definitions, destination):
    identities = set()
    for case in cases:
        validate_fields(case, ("id", "operator", "opset", "inputs", "outputs", "comparison"))
        identity = case["id"]
        directory = case_directory_name(identity)
        if identity.lower() in identities:
            raise ValueError(f"Duplicate case ID (case-insensitive): {identity}")
        identities.add(identity.lower())
        if (
            not isinstance(case["operator"], str)
            or not case["operator"]
            or type(case["opset"]) is not int
            or case["opset"] <= 0
        ):
            raise ValueError("Invalid operator or opset")
        comparison = case["comparison"]
        validate_fields(comparison, ("atol", "rtol", "nan_equal"))
        if any(
            type(comparison[key]) not in (int, float) or not math.isfinite(comparison[key]) or comparison[key] < 0
            for key in ("atol", "rtol")
        ):
            raise ValueError("Invalid comparison tolerance")
        if type(comparison["nan_equal"]) is not bool:
            raise ValueError("Invalid NaN policy")
        if not isinstance(case["inputs"], list) or not isinstance(case["outputs"], list):
            raise ValueError("Inputs and outputs must be arrays")
        inputs = [tensor(value, allow_initializer=True) for value in case["inputs"]]
        outputs = [tensor(value) for value in case["outputs"]]
        if not outputs or len({value.name for value in [*inputs, *outputs]}) != len(inputs) + len(outputs):
            raise ValueError("Missing outputs or duplicate tensor names")
        initializers = [
            value
            for value, definition in zip(inputs, case["inputs"], strict=True)
            if definition.get("initializer", False)
        ]
        feeds = [
            value
            for value, definition in zip(inputs, case["inputs"], strict=True)
            if not definition.get("initializer", False)
        ]
        node = helper.make_node(
            case["operator"], [value.name for value in inputs], [value.name for value in outputs], name="node1"
        )
        graph = helper.make_graph(
            [node],
            identity,
            [value_info(value) for value in feeds],
            [value_info(value) for value in outputs],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", case["opset"])], ir_version=10)
        onnx.checker.check_model(model)
        files = {"model.onnx": model.SerializeToString(deterministic=True)}
        dataset = {"inputs": [], "outputs": []}
        for kind, values in (("inputs", feeds), ("outputs", outputs)):
            for index, value in enumerate(values):
                name = f"test_data_set_0/{kind[:-1]}_{index}.pb"
                files[name] = value.SerializeToString(deterministic=True)
                dataset[kind].append(name)
        manifest = {
            "schema_version": 1,
            "id": identity,
            "model": "model.onnx",
            "required_nodes": ["node1"],
            "datasets": [dataset],
            "comparison": {value.name: comparison for value in outputs},
            "source": {"definition": definitions.name},
        }
        files["case.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        for relative, contents in files.items():
            target = destination / directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)


def identity(definitions):
    sources = [definitions, Path(__file__)]
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
    parser = argparse.ArgumentParser(
        description="Create ONNX models, tensor datasets and case manifests from authored JSON definitions."
    )
    parser.add_argument("--definitions", required=True, type=Path, help="Schema-version-1 JSON definitions file.")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="Fresh directory for directly generated cases.")
    destination.add_argument("--build-cache", type=Path, help="Build directory for digest-keyed reusable cases.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--print-key", action="store_true", help="Print the build-cache fingerprint without generating."
    )
    action.add_argument("--verify", action="store_true", help="Verify existing --build-cache inputs and case bytes.")
    arguments = parser.parse_args()
    if arguments.print_key:
        if arguments.output or arguments.build_cache:
            parser.error("--print-key does not take an output directory")
        print(hashlib.sha256(json.dumps(identity(arguments.definitions), sort_keys=True).encode()).hexdigest())
    elif arguments.verify:
        if arguments.build_cache is None:
            parser.error("--verify requires --build-cache")
        verify_generated(arguments.definitions, arguments.build_cache)
    elif arguments.build_cache:
        materialize(arguments.definitions, arguments.build_cache)
    elif arguments.output:
        generate(arguments.definitions, arguments.output)
    else:
        parser.error("--output or --build-cache is required unless --print-key is set")
