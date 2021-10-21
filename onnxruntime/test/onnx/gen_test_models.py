#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
import onnx
import numpy as np
import os
import argparse
from datetime import date
from onnx import numpy_helper
from onnx import helper
from onnx import utils
from onnx import AttributeProto, TensorProto, GraphProto

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Path to the build directory.")
    return parser.parse_args()


def write_config(model_dir):
    with open(os.path.join(model_dir, "config.txt"), "w") as f:
        f.write("per_sample_tolerance:1e-3\n")
        f.write("relative_per_sample_tolerance:1e-6\n")


def write_tensor(f, c, input_name=None):
    tensor = numpy_helper.from_array(c)
    if input_name:
        tensor.name = input_name
    body = tensor.SerializeToString()
    f.write(body)

def infer_shapes(model_def):
    onnx.checker.check_model(model_def)
    onnx.helper.strip_doc_string(model_def)
    final_model = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(final_model)
    return final_model

def generate_abs_op_test(type, X, top_test_folder):
    for is_raw in [True, False]:
        if is_raw:
            test_folder = os.path.join(top_test_folder, "raw")
        else:
            test_folder = os.path.join(top_test_folder, "not_raw")
        data_dir = os.path.join(test_folder, "test_data_0")
        os.makedirs(data_dir, exist_ok=True)
        # Create one output (ValueInfoProto)
        Y = helper.make_tensor_value_info('Y', type, X.shape)
        X_INFO = helper.make_tensor_value_info('X', type, X.shape)
        if is_raw:
            tensor_x = onnx.helper.make_tensor(name='X', data_type=type, dims=X.shape, vals=X.tobytes(), raw=True)
        else:
            tensor_x = onnx.helper.make_tensor(name='X', data_type=type, dims=X.shape, vals=X.ravel(), raw=False)
        # Create a node (NodeProto)
        node_def = helper.make_node('Abs', inputs=['X'], outputs=['Y'])

        # Create the graph (GraphProto)
        graph_def = helper.make_graph([node_def], 'test-model', [X_INFO], [Y], [tensor_x])
        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name='onnx-example')
        #final_model = infer_shapes(model_def)
        final_model = model_def
        if is_raw:
            onnx.external_data_helper.convert_model_to_external_data(final_model, True)
        onnx.save(final_model, os.path.join(test_folder, 'model.onnx'))
        expected_output_array = np.abs(X)
        expected_output_tensor = numpy_helper.from_array(expected_output_array)
        with open(os.path.join(data_dir, "output_0.pb"), "wb") as f:
            f.write(expected_output_tensor.SerializeToString())


def generate_size_op_test(type, X, test_folder):
    data_dir = os.path.join(test_folder, "test_data_0")
    os.makedirs(data_dir, exist_ok=True)
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [])
    X_INFO = helper.make_tensor_value_info('X', type, X.shape)
    tensor_x = onnx.helper.make_tensor(name='X', data_type=type, dims=X.shape, vals=X.ravel(), raw=False)
    # Create a node (NodeProto)
    node_def = helper.make_node('Size', inputs=['X'], outputs=['Y'])

    # Create the graph (GraphProto)
    graph_def = helper.make_graph([node_def], 'test-model', [X_INFO], [Y], [tensor_x])
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    final_model = infer_shapes(model_def)
    onnx.save(final_model, os.path.join(test_folder, 'model.onnx'))
    expected_output_array = np.int64(X.size)
    expected_output_tensor = numpy_helper.from_array(expected_output_array)
    with open(os.path.join(data_dir, "output_0.pb"), "wb") as f:
        f.write(expected_output_tensor.SerializeToString())


def generate_reducesum_op_test(X, test_folder):
    type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[X.dtype]
    data_dir = os.path.join(test_folder, "test_data_0")
    os.makedirs(data_dir, exist_ok=True)
    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', type, [])
    X_INFO = helper.make_tensor_value_info('X', type, X.shape)
    tensor_x = onnx.helper.make_tensor(name='X', data_type=type, dims=X.shape, vals=X.ravel(), raw=False)
    # Create a node (NodeProto)
    node_def = helper.make_node('ReduceSum', inputs=['X'], outputs=['Y'], keepdims=0)

    # Create the graph (GraphProto)
    graph_def = helper.make_graph([node_def], 'test-model', [X_INFO], [Y], [tensor_x])
    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    final_model = infer_shapes(model_def)
    onnx.save(final_model, os.path.join(test_folder, 'model.onnx'))
    expected_output_array = np.sum(X)
    expected_output_tensor = numpy_helper.from_array(expected_output_array)
    with open(os.path.join(data_dir, "output_0.pb"), "wb") as f:
        f.write(expected_output_tensor.SerializeToString())


def test_abs(output_dir):
    generate_abs_op_test(TensorProto.FLOAT,
                         np.random.randn(3, 4, 5).astype(np.float32), os.path.join(output_dir, 'test_abs_float'))
    generate_abs_op_test(TensorProto.DOUBLE,
                         np.random.randn(3, 4, 5).astype(np.float64), os.path.join(output_dir, 'test_abs_double'))
    generate_abs_op_test(TensorProto.INT8, np.int8([-127, -4, 0, 3, 127]), os.path.join(output_dir, 'test_abs_int8'))
    generate_abs_op_test(TensorProto.UINT8, np.uint8([0, 1, 20, 255]), os.path.join(output_dir, 'test_abs_uint8'))
    generate_abs_op_test(TensorProto.INT16, np.int16([-32767, -4, 0, 3, 32767]),
                         os.path.join(output_dir, 'test_abs_int16'))
    generate_abs_op_test(TensorProto.UINT16, np.uint16([-32767, -4, 0, 3, 32767]),
                         os.path.join(output_dir, 'test_abs_uint16'))
    generate_abs_op_test(TensorProto.INT32, np.int32([-2147483647, -4, 0, 3, 2147483647]),
                         os.path.join(output_dir, 'test_abs_int32'))
    generate_abs_op_test(TensorProto.UINT32, np.uint32([0, 1, 20, 4294967295]),
                         os.path.join(output_dir, 'test_abs_uint32'))
    number_info = np.iinfo(np.int64)
    generate_abs_op_test(TensorProto.INT64, np.int64([-number_info.max, -4, 0, 3, number_info.max]),
                         os.path.join(output_dir, 'test_abs_int64'))
    number_info = np.iinfo(np.uint64)
    generate_abs_op_test(TensorProto.UINT64, np.uint64([0, 1, 20, number_info.max]),
                         os.path.join(output_dir, 'test_abs_uint64'))


def test_reducesum(output_dir):
    generate_reducesum_op_test(
        np.random.randn(3, 4, 5).astype(np.float32), os.path.join(output_dir, 'test_reducesum_random'))


def test_size(output_dir):
    generate_size_op_test(TensorProto.FLOAT,
                          np.random.randn(100, 3000, 10).astype(np.float32),
                          os.path.join(output_dir, 'test_size_float'))
    generate_size_op_test(TensorProto.STRING, np.array(['abc', 'xy'], dtype=np.bytes_),
                          os.path.join(output_dir, 'test_size_string'))


args = parse_arguments()
os.makedirs(args.output_dir, exist_ok=True)
# make test values deterministic but variable
today = date.today()
np.random.seed(today.year + today.month + today.day)
test_abs(args.output_dir)
test_size(args.output_dir)
test_reducesum(args.output_dir)

