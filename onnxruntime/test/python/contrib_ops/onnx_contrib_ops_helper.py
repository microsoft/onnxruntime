# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Helper functions for generating ONNX model and data to test ONNX Runtime contrib ops

import os
import shutil
import subprocess

import onnx
from onnx import numpy_helper

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TOP_DIR, "..", "testdata/")


def prepare_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def _extract_value_info(arr, name, ele_type=None):
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=ele_type if ele_type else onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape,
    )


def generate_data(graph, inputs, outputs, name):
    output_dir = os.path.join(DATA_DIR, name)
    prepare_dir(output_dir)
    model = onnx.helper.make_model(graph)
    with open(os.path.join(output_dir, "model.onnx"), "wb") as f:
        f.write(model.SerializeToString())
    data_set = os.path.join(output_dir, "test_data_set_0")
    prepare_dir(data_set)
    for j, input_np in enumerate(inputs):
        tensor = numpy_helper.from_array(input_np, model.graph.input[j].name)
        with open(os.path.join(data_set, "input_{}.pb".format(j)), "wb") as f:
            f.write(tensor.SerializeToString())
    for j, output_np in enumerate(outputs):
        tensor = numpy_helper.from_array(output_np, model.graph.output[j].name)
        with open(os.path.join(data_set, "output_{}.pb".format(j)), "wb") as f:
            f.write(tensor.SerializeToString())


def expect(
    node,  # type: onnx.NodeProto
    inputs,
    outputs,
    name,
    **kwargs,
):  # type: (...) -> None
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_types = [None] * len(inputs)
    if "input_types" in kwargs:
        input_types = kwargs[str("input_types")]
        del kwargs[str("input_types")]
    output_types = [None] * len(outputs)
    if "output_types" in kwargs:
        output_types = kwargs[str("output_types")]
        del kwargs[str("output_types")]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_types)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(outputs, present_outputs, output_types)
    ]
    graph = onnx.helper.make_graph(nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi)

    generate_data(graph, inputs, outputs, name)

    cwd = os.getcwd()
    onnx_test_runner = os.path.join(cwd, "onnx_test_runner")
    subprocess.run([onnx_test_runner, DATA_DIR + name], check=True, cwd=cwd)
