import numpy as np
from numpy.testing import assert_array_equal
import onnxruntime as ort
import onnx
from onnx import helper
from onnxruntime.nuphar.node_factory import ensure_opset
from onnxruntime.nuphar.model_editor import convert_loop_to_scan_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference, get_shape_from_type_proto
import onnxruntime.tools.onnxruntime_test as ort_test
import argparse
import copy
import os

def save_model_sub_graph(input_model, output_folder):
    def save_subgraph(node, out_mp, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        out_mp.graph.CopyFrom(node.attribute[0].g)
        onnx.save(out_mp, os.path.join(out_path, node.name + '_subgraph.onnx'))

    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.ir_version = 5 # update ir version to avoid requirement of initializer in graph input
    ensure_opset(out_mp, 9) # bump up to ONNX opset 9, which is required for Scan
    out_mp.graph.ClearField('node')
    for in_n in in_mp.graph.node:
        if in_n.op_type in ["Scan", "Loop"]:
            save_subgraph(in_n, copy.deepcopy(out_mp), output_folder)

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
        if node.op_type == 'Loop':
            # for debugging to make scan output as model graph output
            set_op_output_as_model_output(node, model.graph)

def run_with_ort(model_path, symbolic_dims={}, feeds=None, ort_test_case_dir=None):
    def save_tensor_proto(file_path, tp_name, data, data_type=np.float32):
        tp = onnx.TensorProto()
        tp.name = tp_name

        shape = np.shape(data)
        for i in range(0, len(shape)):
            tp.dims.append(shape[i])

        if data_type == np.float32:
            tp.data_type = onnx.TensorProto.FLOAT
            tp.raw_data = data.tobytes()
        elif data_type == np.int64:
            tp.data_type = onnx.TensorProto.INT64
            data=data.astype(np.int64)
            tp.raw_data = data.tobytes()
        elif data_type == np.int:
            tp.data_type = onnx.TensorProto.INT32
            data=data.astype(np.int)
            tp.raw_data = data.tobytes()

        with open(file_path, 'wb') as f:
            f.write(tp.SerializeToString())

    _, feeds, outputs = ort_test.run_model(model_path, symbolic_dims=symbolic_dims, feeds=feeds)

    if ort_test_case_dir:
        model = onnx.load(model_path)
        def save_ort_test_case(ort_test_case_dir):
            for i, (input_name, input) in enumerate(feeds.items()):
                save_tensor_proto(os.path.join(test_data_dir, 'input_{0}.pb'.format(i)),
                                input_name, input)

            output_names = [output.name for output in model.graph.output]
            output_dict = dict(zip(output_names, outputs))
            for i, (output_name, output) in enumerate(output_dict.items()):
                save_tensor_proto(os.path.join(test_data_dir, 'output_{0}.pb'.format(i)), output_name, output)

        save_ort_test_case(ort_test_case_dir)

    return feeds, outputs

def validate_with_ort(input_filename, output_filename, symbolic_dims={}):
    feeds, loop_output = run_with_ort(input_filename, symbolic_dims=symbolic_dims)
    _, scan_output = run_with_ort(output_filename, symbolic_dims=symbolic_dims, feeds=feeds)

    assert(len(loop_output) == len(scan_output))
    for index in range(0, len(loop_output)):
        assert_array_equal(loop_output[index], scan_output[index])

def convert_loop_to_scan_and_validate(input_filename, output_filename, symbolic_dims={}):
    convert_loop_to_scan_model(args.input, args.output)
    validate_with_ort(args.input, args.output, symbolic_dims)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', help='what to do',
                        choices=['save_model_sub_graph',
                                 'run_shape_inference',
                                 'run_with_ort',
                                 'validate_with_ort',
                                 'convert_loop_to_scan_and_validate'])

    parser.add_argument('--input', help='The input model file', default=None)
    parser.add_argument('--output', help='The output model file', default=None)
    parser.add_argument('--symbolic_dims', default={}, type=lambda s: dict(x.split("=") for x in s.split(",")),
                    help='Comma separated name=value pairs for any symbolic dimensions in the model input. '
                            'e.g. --symbolic_dims batch=1,seqlen=5. '
                            'If not provided, the value of 1 will be used for all symbolic dimensions.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.tool == 'save_model_sub_graph':
        save_model_sub_graph(args.input, args.output)
    elif args.tool == 'run_shape_inference':
        run_shape_inference(args.input, args.output)
    elif args.tool == 'run_with_ort':
        run_with_ort(args.input, args.symbolic_dims)
    elif args.tool == 'validate_with_ort':
        validate_with_ort(args.input, args.output, args.symbolic_dims)
    elif args.tool == 'convert_loop_to_scan_and_validate':
        convert_loop_to_scan_and_validate(args.input, args.output, args.symbolic_dims)
