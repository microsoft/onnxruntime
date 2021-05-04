# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import sys
import typing

from util.ort_format_model.types import FbsTypeInfo
# the import of FbsTypeInfo sets up the path so we can import ort_flatbuffers_py
import ort_flatbuffers_py.experimental.fbs as fbs


class OrtFormatModelDumper:
    'Class to dump an ORT format model.'

    def __init__(self, model_path: str):
        '''
        Initialize ORT format model dumper
        :param model_path: Path to model
        '''
        self._file = open(model_path, 'rb').read()
        self._buffer = bytearray(self._file)
        if not fbs.InferenceSession.InferenceSession.InferenceSessionBufferHasIdentifier(self._buffer, 0):
            raise RuntimeError("File does not appear to be a valid ORT format model: '{}'".format(model_path))
        self._model = fbs.InferenceSession.InferenceSession.GetRootAsInferenceSession(self._buffer, 0).Model()

    def _dump_initializers(self, graph: fbs.Graph):
        print('Initializers:')
        for idx in range(0, graph.InitializersLength()):
            tensor = graph.Initializers(idx)
            dims = []
            for dim in range(0, tensor.DimsLength()):
                dims.append(tensor.Dims(dim))

            print(f'{tensor.Name().decode()} data_type={tensor.DataType()} dims={dims}')
        print('--------')

    def _dump_nodeargs(self, graph: fbs.Graph):
        print('NodeArgs:')
        for idx in range(0, graph.NodeArgsLength()):
            node_arg = graph.NodeArgs(idx)
            type = node_arg.Type()
            if not type:
                # NodeArg for optional value that does not exist
                continue

            type_str = FbsTypeInfo.typeinfo_to_str(type)
            value_type = type.ValueType()
            value = type.Value()
            dims = None
            if value_type == fbs.TypeInfoValue.TypeInfoValue.tensor_type:
                tensor_type_and_shape = fbs.TensorTypeAndShape.TensorTypeAndShape()
                tensor_type_and_shape.Init(value.Bytes, value.Pos)
                shape = tensor_type_and_shape.Shape()
                if shape:
                    dims = []
                    for dim in range(0, shape.DimLength()):
                        d = shape.Dim(dim).Value()
                        if d.DimType() == fbs.DimensionValueType.DimensionValueType.VALUE:
                            dims.append(str(d.DimValue()))
                        elif d.DimType() == fbs.DimensionValueType.DimensionValueType.PARAM:
                            dims.append(d.DimParam().decode())
                        else:
                            dims.append('?')
            else:
                dims = None

            print(f'{node_arg.Name().decode()} type={type_str} dims={dims}')
        print('--------')

    def _dump_node(self, node: fbs.Node):
        optype = node.OpType().decode()
        domain = node.Domain().decode() or 'ai.onnx'  # empty domain defaults to ai.onnx

        inputs = [node.Inputs(i).decode() for i in range(0, node.InputsLength())]
        outputs = [node.Outputs(i).decode() for i in range(0, node.OutputsLength())]
        print(f'{node.Index()}:{node.Name().decode()}({domain}:{optype}) '
              f'inputs=[{",".join(inputs)} outputs=[{",".join(outputs)}]')

    def _dump_graph(self, graph: fbs.Graph):
        '''
        Process one level of the Graph, descending into any subgraphs when they are found
        '''

        self._dump_initializers(graph)
        self._dump_nodeargs(graph)
        print('Nodes:')
        for i in range(0, graph.NodesLength()):
            node = graph.Nodes(i)
            self._dump_node(node)

            # Read all the attributes
            for j in range(0, node.AttributesLength()):
                attr = node.Attributes(j)
                attr_type = attr.Type()
                if attr_type == fbs.AttributeType.AttributeType.GRAPH:
                    print(f'## Subgraph for {node.OpType().decode()}.{attr.Name().decode()} ##')
                    self._dump_graph(attr.G())
                    print(f'## End {node.OpType().decode()}.{attr.Name().decode()} Subgraph ##')
                elif attr_type == fbs.AttributeType.AttributeType.GRAPHS:
                    # the ONNX spec doesn't currently define any operators that have multiple graphs in an attribute
                    # so entering this 'elif' isn't currently possible
                    print(f'## Subgraphs for {node.OpType().decode()}.{attr.Name().decode()} ##')
                    for k in range(0, attr.GraphsLength()):
                        print(f'## Subgraph {k} ##')
                        self._dump_graph(attr.Graphs(k))
                        print(f'## End Subgraph {k} ##')

    def dump(self, output: typing.IO):
        graph = self._model.Graph()

        original_stdout = sys.stdout
        sys.stdout = output
        self._dump_graph(graph)
        sys.stdout = original_stdout


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='Dump an ORT format model. Output is to <model_path>.txt')
    parser.add_argument('--stdout', action='store_true', help='Dump to stdout instead of writing to file.')
    parser.add_argument('model_path', help='Path to ORT format model')
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        parser.error(f'{args.model_path} is not a file.')

    return args


def main():
    args = parse_args()
    d = OrtFormatModelDumper(args.model_path)

    if args.stdout:
        d.dump(sys.stdout)
    else:
        output_filename = args.model_path + ".txt"
        with open(output_filename, "w", encoding="utf-8") as ofile:
            d.dump(ofile)


if __name__ == '__main__':
    main()
