# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import onnx

from ...fusions import FusionGelu, FusionLayerNormalization
from ...onnx_model import ONNXModel
from .fusion_lpnorm import FusionLpNormalization


def qnn_preprocess_model(
    model_input: Path,
    model_output: Path,
    fuse_layernorm: bool = False,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = False,
    external_data_location: str | None = None,
    external_data_size_threshold: int = 1024,
    external_data_convert_attribute: bool = False,
) -> bool:
    """
    If necessary, this method creates a new "pre-processed" model in preparation for
    quantization of a model to be used in QNN EP. Returns true if a new model was created.

    This method perfoms the following operations:
    - Fuse Erf sequence into a single Gelu node.
    - Fuse ReduceL2 sequence into a single LpNormalization node (p == 2).
    - (Optional) Fuse ReduceMean sequence into a single LayerNormalization node.

    Args:
        model_input: Path to the input model file.
        model_output: Path the output model file, which is only created if this method returns True.
        fuse_layernorm: True if ReduceMean sequences should be fused into LayerNormalization nodes.
            Defaults to False.
        save_as_external_data: True if output model should be saved with external data. Defaults to false.
        all_tensors_to_one_file: Effective only if save_as_external_data is true. Defaults to false.
            If true, save all tensors to one external file specified by external_data_location.
            If false, save each tensor to a file named with the tensor name.
        external_data_location: Effective only if save_as_external_data is true. Defaults to None.
            Specify the external file to which all tensors are saved. Path is relative
            to the model path. If not specified, the model's name is used.
        external_data_size_threshold: Effective only if save_as_external_data is true. Defaults to 1024.
            Tensors with a data size >= external_data_size_threshold are converted to external data.
            To convert every tensor with raw data to external data, set to 0.
        external_data_convert_attribute: Effective only if save_as_external_data is true. Defaults to false.
            If true, convert all tensors to external data.
            If false, convert only non-attribute tensors to external data.
    """
    modified = False
    model = onnx.load_model(model_input)
    onnx_model = ONNXModel(model)

    # Fuse Erf sequence into a single Gelu
    fusion_gelu = FusionGelu(onnx_model)
    if fusion_gelu.apply():
        modified = True

    # Fuse ReduceL2 sequence into a single LpNormalization node with p == 2.
    fusion_lpnorm = FusionLpNormalization(onnx_model)
    if fusion_lpnorm.apply():
        modified = True

    # Optionally, fuse ReduceMean sequence into a single LayerNormalization node.
    if fuse_layernorm:
        onnx_opset = next(x for x in model.opset_import if x.domain == "" or x.domain == "ai.onnx")

        # Need opset >= 17 to use LayerNormalization.
        if onnx_opset.version < 17:
            logging.warning(
                "Unable to fuse ReduceMean sequence into a LayerNormalization node. "
                "ONNX model must use an opset >= 17 in order to use LayerNormalization, "
                f"but found version {onnx_opset.version}. Please use onnx.version_converter to update your model."
            )
        else:
            fusion_layernorm = FusionLayerNormalization(onnx_model)
            if fusion_layernorm.apply():
                modified = True

    # Make sure all nodes have a name.
    unnamed_node_prefix = "qnn_preproc_node_"
    available_suffix = onnx_model.get_largest_node_name_suffix(unnamed_node_prefix) + 1
    for node in onnx_model.model.graph.node:
        if node.op_type != "Constant" and not node.name:
            new_node_name = f"{unnamed_node_prefix}{available_suffix!s}"
            available_suffix += 1
            node.name = new_node_name
            modified = True
            logging.warning(f"Node of type {node.op_type} does not have a name. Renamed to {new_node_name}.")

    if modified:
        onnx_model.topological_sort()
        onnx.save_model(
            model,
            model_output,
            save_as_external_data=save_as_external_data,
            all_tensors_to_one_file=all_tensors_to_one_file,
            location=external_data_location,
            size_threshold=external_data_size_threshold,
            convert_attribute=external_data_convert_attribute,
        )

    return modified
