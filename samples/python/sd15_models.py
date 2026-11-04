# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Model-only SD1.5 boundary and safety preprocessing adapters."""

import hashlib
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from stateful_demo_assets import digest


def initializer_fingerprints(model) -> dict[str, str]:
    return {value.name: hashlib.sha256(value.SerializeToString()).hexdigest() for value in model.graph.initializer}


def boundary_model(
    source: Path, destination: Path, *, latent_scale: bool = False, independent_image_axes: bool = False
) -> dict:
    """Preserve original tensors while exposing FP32 boundaries around FP16 compute."""
    model = onnx.load(source, load_external_data=False)
    original = initializer_fingerprints(model)
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    names.update(
        value.name
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info, *model.graph.initializer)
    )
    if any(name.startswith("__sd15_") for name in names):
        raise ValueError("Reserved SD1.5 boundary name collides with an original value")
    if independent_image_axes:
        image = next((value for value in model.graph.input if value.name == "images"), None)
        if image is None or len(image.type.tensor_type.shape.dim) != 4:
            raise ValueError("Safety image axes require a rank-4 images input")
        # The source incorrectly equates 512px image axes with 224px CLIP axes.
        for index, name in ((1, "__sd15_image_height"), (2, "__sd15_image_width")):
            dimension = image.type.tensor_type.shape.dim[index]
            if dimension.HasField("dim_param"):
                dimension.dim_param = name
    before, after = [], []
    for value in model.graph.input:
        if value.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        external = value.name
        internal = f"__sd15_input_{external}"
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == external:
                    node.input[index] = internal
        for info in model.graph.value_info:
            if info.name == external:
                info.name = internal
        value.type.tensor_type.elem_type = TensorProto.FLOAT
        cast_source = external
        if latent_scale and external == "latent_sample":
            constant = "__sd15_inverse_vae_scale"
            model.graph.initializer.append(numpy_helper.from_array(np.array(1 / 0.18215, dtype=np.float32), constant))
            cast_source = "__sd15_scaled_latents"
            before.append(helper.make_node("Mul", [external, constant], [cast_source]))
        before.append(helper.make_node("Cast", [cast_source], [internal], to=TensorProto.FLOAT16))
    for value in model.graph.output:
        if value.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        external = value.name
        internal = f"__sd15_output_{external}"
        for node in model.graph.node:
            for values in (node.input, node.output):
                for index, name in enumerate(values):
                    if name == external:
                        values[index] = internal
        for info in model.graph.value_info:
            if info.name == external:
                info.name = internal
        value.type.tensor_type.elem_type = TensorProto.FLOAT
        after.append(helper.make_node("Cast", [internal], [external], to=TensorProto.FLOAT))
    nodes = [*before, *model.graph.node, *after]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if any(node.domain == "com.microsoft" for node in model.graph.node) and not any(
        entry.domain == "com.microsoft" for entry in model.opset_import
    ):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    current = initializer_fingerprints(model)
    if any(current.get(name) != digest for name, digest in original.items()):
        raise ValueError("Original model initializer changed during boundary adaptation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".boundary-", dir=destination.parent) as temporary:
        pending = Path(temporary) / "model.onnx"
        onnx.save_model(model, pending)
        pending.replace(destination)
    return {
        "original_initializers": len(original),
        "preserved": True,
        "latent_scale": latent_scale,
        "independent_image_axes": independent_image_axes,
        "inputs": len(before),
        "outputs": len(after),
        "sha256": digest(destination),
    }


def bicubic_coefficients(source: int, target: int) -> np.ndarray:
    """Fixed-point two-pass bicubic coefficients for the pinned Pillow RGB convention."""
    if type(source) is not int or type(target) is not int or source <= 0 or target <= 0:
        raise ValueError("Resize dimensions must be positive integers")
    ratio = source / target
    scale = max(ratio, 1)
    support = 2 * scale
    matrix = np.zeros((source, target), dtype=np.float64)
    for column in range(target):
        center = (column + 0.5) * ratio
        left = max(0, int(center - support + 0.5))
        right = min(source, int(center + support + 0.5))
        distance = np.abs((np.arange(left, right) - center + 0.5) / scale)
        a = -0.5
        weights = np.where(
            distance < 1,
            ((a + 2) * distance - (a + 3)) * distance * distance + 1,
            np.where(distance < 2, ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a, 0),
        )
        weights /= math.fsum(weights)
        scaled = weights * (1 << 22)
        matrix[left:right, column] = np.trunc(scaled + np.where(scaled < 0, -0.5, 0.5))
    return matrix


def safety_preprocess_model(destination: Path, *, image_size: int = 512, crop_size: int = 224):
    """Keep exact uint8 bicubic rounding inside an ordinary CPU ONNX Session."""
    weights = bicubic_coefficients(image_size, crop_size)
    tensors = {
        "half": np.array(0.5, dtype=np.float32),
        "zero": np.array(0, dtype=np.float32),
        "one": np.array(1, dtype=np.float32),
        "byte_scale": np.array(255, dtype=np.float32),
        "weights_h": weights,
        "weights_v": weights.T.copy(),
        "round_bias": np.array(1 << 21, dtype=np.float64),
        "fixed_scale": np.array(1 << 22, dtype=np.float64),
        "zero64": np.array(0, dtype=np.float64),
        "max64": np.array(255, dtype=np.float64),
        "rescale": np.array(1 / 255, dtype=np.float64),
        "mean": np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 3, 1, 1),
        "std": np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 3, 1, 1),
    }
    nodes = [
        helper.make_node("Mul", ["sample", "half"], ["half_image"]),
        helper.make_node("Add", ["half_image", "half"], ["unclipped"]),
        helper.make_node("Clip", ["unclipped", "zero", "one"], ["normalized"]),
        helper.make_node("Transpose", ["normalized"], ["images"], perm=[0, 2, 3, 1]),
        helper.make_node("Mul", ["normalized", "byte_scale"], ["byte_float"]),
        helper.make_node("Round", ["byte_float"], ["rounded_bytes"]),
        helper.make_node("Cast", ["rounded_bytes"], ["pixels64"], to=TensorProto.DOUBLE),
        helper.make_node("MatMul", ["pixels64", "weights_h"], ["horizontal"]),
    ]
    for prefix, source in (("h", "horizontal"), ("v", "vertical")):
        nodes.extend(
            [
                helper.make_node("Add", [source, "round_bias"], [prefix + "_biased"]),
                helper.make_node("Div", [prefix + "_biased", "fixed_scale"], [prefix + "_scaled"]),
                helper.make_node("Floor", [prefix + "_scaled"], [prefix + "_integer"]),
                helper.make_node("Clip", [prefix + "_integer", "zero64", "max64"], [prefix + "_pixels"]),
            ]
        )
        if prefix == "h":
            nodes.append(helper.make_node("MatMul", ["weights_v", "h_pixels"], ["vertical"]))
    nodes.extend(
        [
            helper.make_node("Mul", ["v_pixels", "rescale"], ["rescaled64"]),
            helper.make_node("Cast", ["rescaled64"], ["rescaled"], to=TensorProto.FLOAT),
            helper.make_node("Sub", ["rescaled", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["clip_input"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "sd15_safety_preprocessing",
        [helper.make_tensor_value_info("sample", TensorProto.FLOAT, [1, 3, image_size, image_size])],
        [
            helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, image_size, image_size, 3]),
            helper.make_tensor_value_info("clip_input", TensorProto.FLOAT, [1, 3, crop_size, crop_size]),
        ],
        [numpy_helper.from_array(value, name) for name, value in tensors.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    return destination
