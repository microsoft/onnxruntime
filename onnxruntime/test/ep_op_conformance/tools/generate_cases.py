import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(required) - value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Missing or unsupported definition fields")


def tensor(definition, allow_initializer=False):
    fields(definition, ("name", "shape", "values"), ("initializer",) if allow_initializer else ())
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
    document = json.loads(definitions.read_text(encoding="utf-8"))
    fields(document, ("schema_version", "cases"))
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not isinstance(document["cases"], list)
        or not document["cases"]
    ):
        raise ValueError("Unsupported definition version or empty case set")
    if destination.exists():
        raise ValueError("Destination already exists; generated cases are immutable")
    payloads = []
    identities = set()
    for case in document["cases"]:
        fields(case, ("id", "operator", "opset", "inputs", "outputs", "comparison"))
        identity = case["id"]
        if (
            not isinstance(identity, str)
            or not identity
            or identity in (".", "..")
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in identity)
        ):
            raise ValueError("Invalid case ID")
        if identity in identities:
            raise ValueError("Duplicate case ID")
        identities.add(identity)
        if (
            not isinstance(case["operator"], str)
            or not case["operator"]
            or type(case["opset"]) is not int
            or case["opset"] <= 0
        ):
            raise ValueError("Invalid operator or opset")
        comparison = case["comparison"]
        fields(comparison, ("atol", "rtol", "nan_equal"))
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
            "source": {"definition": definitions.name, "case_id": identity},
        }
        files["case.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        payloads.append((identity, files))
    destination.mkdir(parents=True)
    try:
        for identity, files in payloads:
            for relative, contents in files.items():
                target = destination / identity / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
    except Exception:
        shutil.rmtree(destination)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--definitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.definitions, arguments.output)
