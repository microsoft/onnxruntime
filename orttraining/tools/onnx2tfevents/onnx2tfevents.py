# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This tool is to convert ONNX model to TensorBoard events file so that we can visualize the model in TensorBoard.
# this is especially useful for debugging large models that cannot be visualized in Netron.
#
# Usage: python onnx2tfevents.py --logdir <tensorboard log directory> --model <onnx model path>
#
# Note: This tool requires tensorboard to be installed.

import argparse
import inspect
import itertools
from abc import ABC, abstractmethod

import numpy as np
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.versions_pb2 import VersionDef
from torch.utils.tensorboard import SummaryWriter

import onnx
from onnx import helper, numpy_helper

_DTYPE_ONNX_TO_TF = {
    1: 1,  # FLOAT
    2: 4,  # UINT8
    3: 6,  # INT8
    4: 17,  # UINT16
    5: 5,  # INT16
    6: 3,  # INT32
    7: 9,  # INT64
    8: 7,  # STRING
    9: 10,  # BOOL
    10: 19,  # FLOAT16
    11: 2,  # DOUBLE
    12: 22,  # UINT32
    13: 23,  # UINT64
    14: 8,  # COMPLEX64
    15: 18,  # COMPLEX128
    16: 14,  # BFLOAT16
}


def dtype_onnx_to_tf(onnx_dtype):
    if onnx_dtype in _DTYPE_ONNX_TO_TF:
        return _DTYPE_ONNX_TO_TF[onnx_dtype]
    return 0


def get_node_by_input(graph, input_name):
    for node in graph.node:
        for input in node.input:
            if input == input_name:
                return node
    return None


def get_prefix(name):
    pos = name.rfind("/")
    if pos >= 0:
        return name[: pos + 1]
    return ""


def parse(graph):
    nodes = []

    def _add_io_node(node, type):
        shape_proto = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=d.dim_value, name=d.dim_param) for d in node.type.tensor_type.shape.dim]
        )
        nodes.append(
            NodeDef(
                name=node.name.encode(encoding="utf_8"),
                op=type,
                input=[],
                attr={
                    "dtype": AttrValue(type=dtype_onnx_to_tf(node.type.tensor_type.elem_type)),
                    "shape": AttrValue(shape=shape_proto),
                },
            )
        )

    for node in graph.input:
        _add_io_node(node, "Input")
    for node in graph.output:
        _add_io_node(node, "Output")

    for node in graph.initializer:
        shape_proto = TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in node.dims])
        nodes.append(
            NodeDef(
                name=node.name.encode(encoding="utf_8"),
                op="Const",
                input=[],
                attr={
                    "dtype": AttrValue(type=dtype_onnx_to_tf(node.data_type)),
                    "shape": AttrValue(shape=shape_proto),
                    "attr": AttrValue(
                        s=np.array2string(numpy_helper.to_array(node), separator=",")
                        .replace("\n", "")
                        .replace(" ", "")
                        .encode(encoding="utf_8")
                    ),
                },
            )
        )

    value_info_map = {}
    for node in graph.value_info:
        shape_proto = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=d.dim_value, name=d.dim_param) for d in node.type.tensor_type.shape.dim]
        )
        value_info_map[node.name] = (shape_proto, node.type.tensor_type.elem_type)

    for node in graph.node:
        _attr = []
        for s in node.attribute:
            _attr.append(" = ".join([str(f[1]) for f in s.ListFields()]))
        attr = ", ".join(_attr).encode(encoding="utf_8")
        shape_proto = None
        elem_type = None
        if node.output[0] in value_info_map:
            shape_proto, elem_type = value_info_map[node.output[0]]
        nodes.append(
            NodeDef(
                name=node.output[0].encode(encoding="utf_8"),
                op=node.op_type,
                input=node.input,
                attr={
                    "dtype": AttrValue(type=dtype_onnx_to_tf(elem_type)),
                    "shape": AttrValue(shape=shape_proto),
                    "parameters": AttrValue(s=attr),
                },
            )
        )

    return GraphDef(node=nodes, versions=VersionDef(producer=22))


def add_onnx_model(self, model):
    self._get_file_writer().add_onnx_graph(parse(model.graph))


# Monkey patch SummaryWriter to add onnx graph.
SummaryWriter.add_onnx_model = add_onnx_model


class TransformerBase(ABC):
    """Abstract base class for transformers that need to be applied to the ONNX graph before
    converting it to TensorBoard. This base class is also used as a registry for all non-abstract transformers.
    NOTE: Transformers are applied in the order they are registered.
    """

    _TRANSFORMERS = []

    @classmethod
    def register_transformer(cls, klass):
        if not inspect.isabstract(klass):
            cls._TRANSFORMERS.append(klass())

    @classmethod
    def run_transformers(cls, graph):
        for transformer in cls._TRANSFORMERS:
            transformer.transform(graph)

    def __init_subclass__(cls):
        super().__init_subclass__()
        TransformerBase.register_transformer(cls)

    @abstractmethod
    def transform(self, graph):
        pass

    def _apply_to_all_names(self, graph, func):
        for node in graph.node:
            for idx, name in enumerate(node.input):
                node.input[idx] = func(name)
            for idx, name in enumerate(node.output):
                node.output[idx] = func(name)
        for node in itertools.chain(graph.input, graph.output, graph.initializer, graph.value_info):
            node.name = func(node.name)


class HierarchicalNameTransformer(TransformerBase):
    """TensorBoard can show the graph in different level for better visualization. PyTorch exporter can already
    add hierarchical information to the ONNX graph nodes, but the behavior is not consistent. For example,
    for graph inputs and outputs, the exported name is _original_module.A.B.C, but for activations and initializers,
    the exported name is /_original_module/A/B/C. This transformer will make the exported names consistent
    to use "/" as the separator. The top level _original_module will also be removed.
    """

    def __init__(self):
        super().__init__()
        self.sections = set()
        self.original_module_name = "_original_module"

    def _add_sections(self, name):
        if "/" in name:
            for sec in name.split("/"):
                if len(sec) > 0:
                    self.sections.add(sec)

    def _transform_name(self, name):
        if "/" in name:
            if name.startswith("/" + self.original_module_name + "/"):
                name = name[len(self.original_module_name) + 2 :]
            if name.startswith("/"):
                name = name[1:]
            return name
        new_name = ""
        while True:
            curr_section = ""
            for section in self.sections:
                if name.startswith(section) and (len(name) == len(section) or name[len(section)] == "."):
                    if section != self.original_module_name:
                        new_name = new_name + "/" + section
                    curr_section = section
                    break
            if curr_section == "":
                new_name = new_name + "/" + name
                break
            else:
                if name == curr_section:
                    break
                name = name[len(curr_section) + 1 :]
                assert len(name) > 0
        return new_name[1:]

    def transform(self, graph):
        for node in graph.node:
            for name in itertools.chain(node.input, node.output):
                self._add_sections(name)
        for initializer in graph.initializer:
            self._add_sections(initializer.name)
        self._apply_to_all_names(graph, self._transform_name)


class ReplaceNameTransformer(TransformerBase):
    """Abstract class for transformers that need to replace the names of edges in the ONNX graph.
    The sub-class need to prepare the new names in the _generate_new_names method.
    """

    def __init__(self):
        super().__init__()
        self.new_names = {}

    def _transform_name(self, name):
        if name in self.new_names:
            return self.new_names[name]
        return name

    @abstractmethod
    def _generate_new_names(self, graph):
        pass

    def transform(self, graph):
        self._generate_new_names(graph)
        self._apply_to_all_names(graph, self._transform_name)


class SplitSqueezeTransformer(ReplaceNameTransformer):
    """Handle the edge names without hierarchical information that are introduced by GatherToSplitFusion."""

    def _generate_new_names(self, graph):
        for node in graph.node:
            if node.doc_string == "Split for Fused Gather nodes":
                squeeze_node = get_node_by_input(graph, node.output[0])
                assert squeeze_node is not None
                prefix = get_prefix(squeeze_node.output[0])
                if len(prefix) > 0:
                    for output in node.output:
                        self.new_names[output] = prefix + output
                        self.new_names[output + "_grad"] = prefix + output + "_grad"


class ExtraOutputsTransformer(ReplaceNameTransformer):
    """Handle the output names without hierarchical information that are introduced by some fusions.
    This kind of fusions will add extra outputs to the node, we will try to use the hierarchical information
    from the first output of the node.
    """

    def __init__(self):
        super().__init__()
        self.supported_ops = [
            "LayerNormalization",
            "ConcatTraining",
            "BitmaskDropout",
            "BiasSoftmaxDropout",
            "BitmaskBiasDropout",
        ]

    def _generate_new_names(self, graph):
        for node in graph.node:
            if node.op_type in self.supported_ops and len(node.output) > 1:
                prefix = get_prefix(node.output[0])
                if len(prefix) > 0:
                    for output in node.output[1:]:
                        if "/" not in output:
                            self.new_names[output] = prefix + output


class YieldOpTransformer(ReplaceNameTransformer):
    """Handle the output names without hierarchical information that are introduced by YieldOp."""

    def _generate_new_names(self, graph):
        for node in graph.node:
            if node.op_type == "YieldOp":
                # TODO: the length of input and output can be not equal if attribute non_differentiable_outputs
                # is not empty. Need to handle this case.
                assert len(node.input) == len(node.output)
                for idx in range(len(node.input)):
                    prefix = get_prefix(node.input[idx])
                    if len(prefix) > 0:
                        self.new_names[node.output[idx]] = prefix + node.output[idx]


class ListUnpackTransformer(TransformerBase):
    """Tensorboard's NodeDef doesn't have output, it assumes each node has only one output and put this output as
    node name. If an ONNX node have more than one outputs, we will add a special node named ListUnpack for each output.
    """

    def __init__(self):
        self.ops = {}

    def transform(self, graph):
        new_nodes = []
        for node in graph.node:
            if len([output for output in node.output if len(output) > 0]) > 1:
                idx = 0
                if node.op_type in self.ops:
                    idx = self.ops[node.op_type]
                    self.ops[node.op_type] = idx + 1
                else:
                    self.ops[node.op_type] = 1
                new_output = get_prefix(node.output[0]) + node.op_type + "_" + str(idx) + "_output"
                for output in node.output:
                    if len(output) > 0:
                        new_nodes.append(helper.make_node("ListUnpack", [new_output], [output]))
                node.ClearField("output")
                node.output.extend([new_output])
        if len(new_nodes) > 0:
            graph.node.extend(new_nodes)


class AppendOpTypeTransformer(ReplaceNameTransformer):
    """Tensorboard's node search can only index the node name, not the node type. To make the op type searchable,
    append the op type to the end of the node name.
    """

    def _generate_new_names(self, graph):
        for node in graph.node:
            self.new_names[node.output[0]] = node.output[0] + "::" + node.op_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", help="Tensorboard log directory")
    parser.add_argument("--model", help="ONNX model path")

    args = parser.parse_args()
    if not args.logdir or not args.model:
        print("Fail to convert. Please specify the tensorboard log directory and the onnx model path.")
        print("Usage: python3 onnx2tfevents.py --logdir <tensorboard log directory> --model <onnx model path>")
        return

    model = onnx.load(args.model)
    TransformerBase.run_transformers(model.graph)
    with SummaryWriter(args.logdir) as writer:
        writer.add_onnx_model(model)

    print("Successfully converted the onnx model to tensorboard log directory. Start tensorboard to browse.")


if __name__ == "__main__":
    main()
