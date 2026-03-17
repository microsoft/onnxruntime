# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import onnx
from onnxscript import ir

logger = logging.getLogger(__name__)


def _flatten_annotations(layer_annotations: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten a dict of {layer_name: [substring, ...]} into [(substring, layer_name), ...]."""
    return [
        (substring, layer_name)
        for layer_name, substrings in layer_annotations.items()
        for substring in substrings
    ]


# ---------------------------------------------------------------------------
# ir.Model path (used by conversion.py)
# ---------------------------------------------------------------------------


def _annotate_ir_graph(graph: ir.Graph, substring_annotations: list[tuple[str, str]]) -> None:
    """Annotate nodes in an ir.Graph, recursing into subgraphs."""
    for node in graph:
        if node.name is None:
            continue

        matched_annotation = None
        for substring, annotation in substring_annotations:
            if substring in node.name:
                matched_annotation = annotation

        if matched_annotation is not None:
            node.metadata_props["layer_ann"] = matched_annotation

        # Recurse into subgraphs for control-flow nodes (If, Loop, etc.)
        for attr in node.attributes.values():
            if isinstance(attr, ir.Attr) and attr.type == ir.AttributeType.GRAPH:
                _annotate_ir_graph(attr.value, substring_annotations)
            elif isinstance(attr, ir.Attr) and attr.type == ir.AttributeType.GRAPHS:
                for sub_graph in attr.value:
                    _annotate_ir_graph(sub_graph, substring_annotations)


def annotate_ir_model(model: ir.Model, layer_annotations: dict[str, list[str]]) -> None:
    """Annotate an onnxscript ir.Model with layer annotations.

    For each node whose name contains a configured substring, a metadata property
    ``layer_ann`` is set to the corresponding layer name.  If multiple substrings
    match, the last one in iteration order wins (consistent with the ORT reference
    implementation).

    :param model: The onnxscript IR model to annotate.
    :param layer_annotations: Mapping of layer name to list of node-name substrings.
    """
    substring_annotations = _flatten_annotations(layer_annotations)
    _annotate_ir_graph(model.graph, substring_annotations)


# ---------------------------------------------------------------------------
# onnx.ModelProto path (used by model_builder.py)
# ---------------------------------------------------------------------------


def _annotate_proto_graph(graph: onnx.GraphProto, substring_annotations: list[tuple[str, str]]) -> None:
    """Annotate nodes in an onnx.GraphProto, recursing into subgraphs."""
    for node in graph.node:
        matched_annotation = None
        for substring, annotation in substring_annotations:
            if substring in node.name:
                matched_annotation = annotation

        if matched_annotation is not None:
            entry = None
            for prop in node.metadata_props:
                if prop.key == "layer_ann":
                    entry = prop
                    break

            if entry:
                entry.value = matched_annotation
            else:
                entry = node.metadata_props.add()
                entry.key = "layer_ann"
                entry.value = matched_annotation

        # Recurse into subgraphs for control-flow nodes (If, Loop, etc.)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _annotate_proto_graph(attr.g, substring_annotations)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub_graph in attr.graphs:
                    _annotate_proto_graph(sub_graph, substring_annotations)


def annotate_proto_model(model: onnx.ModelProto, layer_annotations: dict[str, list[str]]) -> None:
    """Annotate an onnx.ModelProto with layer annotations.

    For each node whose name contains a configured substring, a metadata property
    ``layer_ann`` is set to the corresponding layer name.  If multiple substrings
    match, the last one in iteration order wins (consistent with the ORT reference
    implementation).

    :param model: The ONNX ModelProto to annotate.
    :param layer_annotations: Mapping of layer name to list of node-name substrings.
    """
    substring_annotations = _flatten_annotations(layer_annotations)
    _annotate_proto_graph(model.graph, substring_annotations)
