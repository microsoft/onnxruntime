import argparse
import copy
import os

import numpy as np
import onnx
from numpy.testing import assert_array_equal
from onnx import helper

import onnxruntime as ort
import onnxruntime.tools.onnxruntime_test as ort_test
from onnxruntime.nuphar.model_editor import convert_loop_to_scan_model
from onnxruntime.nuphar.node_factory import ensure_opset
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, get_shape_from_type_proto


def run_shape_inference(input_model, output_model):
    in_mp = onnx.load(input_model)
    in_mp = SymbolicShapeInference.infer_shapes(in_mp, auto_merge=True)
    onnx.save(in_mp, output_model)


# use this function to make a loop op's output as model output.
# it helps to debug data issues when edited model outputs do not match the original model.
def extract_loop_outputs_as_model_outputs(model):
    def set_op_output_as_model_output(node, graph):
        for output in node.output:
            for value_info in graph.value_info:
                if value_info.name == output:
                    graph.value_info.remove(value_info)
                    output_value_info = graph.output.add()
                    output_value_info.CopyFrom(value_info)
                    break

    for node in model.graph.node:
        if node.op_type == "Loop":
            # for debugging to make scan output as model graph output
            set_op_output_as_model_output(node, model.graph)


def run_with_ort(model_path, symbolic_dims={}, feeds=None, ort_test_case_dir=None):
    _, feeds, outputs = ort_test.run_model(
        model_path, symbolic_dims=symbolic_dims, feeds=feeds, override_initializers=False
    )

    if ort_test_case_dir:
        model = onnx.load(model_path)

        def save_ort_test_case(ort_test_case_dir):
            if not os.path.exists(ort_test_case_dir):
                os.makedirs(ort_test_case_dir)

            test_data_set_dir = os.path.join(ort_test_case_dir, "test_data_set_0")
            if not os.path.exists(test_data_set_dir):
                os.makedirs(test_data_set_dir)

            onnx.save(model, os.path.join(ort_test_case_dir, "model.onnx"))
            for i, (input_name, input) in enumerate(feeds.items()):
                onnx.save_tensor(
                    onnx.numpy_helper.from_array(input, input_name),
                    os.path.join(test_data_set_dir, "input_{0}.pb".format(i)),
                )

            output_names = [output.name for output in model.graph.output]
            output_dict = dict(zip(output_names, outputs))
            for i, (output_name, output) in enumerate(output_dict.items()):
                onnx.save_tensor(
                    onnx.numpy_helper.from_array(output, output_name),
                    os.path.join(test_data_set_dir, "output_{0}.pb".format(i)),
                )

        save_ort_test_case(ort_test_case_dir)

    return feeds, outputs


def validate_with_ort(input_filename, output_filename, symbolic_dims={}):
    feeds, loop_output = run_with_ort(input_filename, symbolic_dims=symbolic_dims)
    _, scan_output = run_with_ort(output_filename, symbolic_dims=symbolic_dims, feeds=feeds)

    assert len(loop_output) == len(scan_output)
    for index in range(0, len(loop_output)):
        assert_array_equal(loop_output[index], scan_output[index])


def convert_loop_to_scan_and_validate(input_filename, output_filename, symbolic_dims={}):
    convert_loop_to_scan_model(args.input, args.output)
    validate_with_ort(args.input, args.output, symbolic_dims=symbolic_dims)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool",
        help="what to do",
        choices=["run_shape_inference", "run_with_ort", "validate_with_ort", "convert_loop_to_scan_and_validate"],
    )

    parser.add_argument("--input", help="The input model file", default=None)
    parser.add_argument("--output", help="The output model file", default=None)
    parser.add_argument(
        "--symbolic_dims",
        default={},
        type=lambda s: dict(x.split("=") for x in s.split(",")),
        help="Comma separated name=value pairs for any symbolic dimensions in the model input. "
        "e.g. --symbolic_dims batch=1,seqlen=5. "
        "If not provided, the value of 1 will be used for all symbolic dimensions.",
    )
    parser.add_argument("--ort_test_case_dir", help="ort test case dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.tool == "run_shape_inference":
        run_shape_inference(args.input, args.output)
    elif args.tool == "run_with_ort":
        run_with_ort(args.input, symbolic_dims=args.symbolic_dims, ort_test_case_dir=args.ort_test_case_dir)
    elif args.tool == "validate_with_ort":
        validate_with_ort(args.input, args.output, symbolic_dims=args.symbolic_dims)
    elif args.tool == "convert_loop_to_scan_and_validate":
        convert_loop_to_scan_and_validate(args.input, args.output, symbolic_dims=args.symbolic_dims)
