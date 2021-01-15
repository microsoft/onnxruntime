import os
import argparse
import json
import onnx
import psutil
import numpy
"""
This profiler tool could run a transformer model and print out the kernel time spent on each Node of the model.
Example of profiling of longformer model:
    python profiler.py --model longformer-base-4096_fp32.onnx --batch_size 1 --sequence_length 4096 --global_length 8 --samples 1000 --thread_num 8 --dummy_inputs longformer --use_gpu
"""

NODES_TYPE_CONTAINING_SUBGRAPH = ['Scan', 'Loop', 'If']


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str, help="onnx model path")

    parser.add_argument('-b', '--batch_size', required=False, type=int, default=1, help="batch size of input")

    parser.add_argument('-s',
                        '--sequence_length',
                        required=False,
                        type=int,
                        default=32,
                        help="sequence length of input")

    parser.add_argument('--past_sequence_length',
                        required=False,
                        type=int,
                        default=1,
                        help="past sequence length for gpt2")

    parser.add_argument('--global_length',
                        required=False,
                        type=int,
                        default=1,
                        help="number of global tokens for longformer")

    parser.add_argument(
        '--samples',
        required=False,
        type=int,
        default=1000,
        help="number of samples to test. Set it large enough to reduce the variance of performance result.")

    parser.add_argument(
        '--threshold',
        required=False,
        type=float,
        default=0,
        help=
        "Threshold of ratio of run time of a node among all nodes. Nodes that nodes with lower ratio will not be in detail results."
    )

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="number of threads to use")

    parser.add_argument('--input_ids_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for input ids, for bert")
    parser.add_argument('--segment_ids_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for segment ids, for bert")
    parser.add_argument('--input_mask_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for attention mask, for bert")

    parser.add_argument('--dummy_inputs',
                        required=False,
                        default='default',
                        choices=['bert', 'gpt2', 'longformer', 'default'],
                        help="Type of dummy inputs. The default will create inputs with ones.")

    parser.add_argument('-g', '--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        '--basic_optimization',
        required=False,
        action='store_true',
        help="Enable only basic graph optimizations. By default, all optimizations are enabled in OnnxRuntime")
    parser.set_defaults(basic_optimization=False)

    parser.add_argument('--kernel_time_only',
                        required=False,
                        action='store_true',
                        help="Only include the kernel time and no fence time")
    parser.set_defaults(kernel_time_only=False)

    parser.add_argument('-v', '--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    return parser.parse_args(argv)


def create_bert_inputs(model, batch_size, sequence_length, samples, input_ids_name, segment_ids_name, input_mask_name):
    from bert_test_data import get_bert_inputs, generate_test_data
    input_ids, segment_ids, input_mask = get_bert_inputs(model, input_ids_name, segment_ids_name, input_mask_name)
    all_inputs = generate_test_data(batch_size,
                                    sequence_length,
                                    test_cases=samples,
                                    seed=123,
                                    verbose=False,
                                    input_ids=input_ids,
                                    segment_ids=segment_ids,
                                    input_mask=input_mask,
                                    random_mask_length=False)

    return all_inputs


def run_profile(onnx_model_path, use_gpu, basic_optimization, thread_num, all_inputs):
    from benchmark_helper import create_onnxruntime_session

    session = create_onnxruntime_session(onnx_model_path,
                                         use_gpu,
                                         enable_all_optimization=not basic_optimization,
                                         num_threads=thread_num,
                                         enable_profiling=True)

    for inputs in all_inputs:
        _ = session.run(None, inputs)

    profile_file = session.end_profiling()
    return profile_file


def load_profile_json(profile_file):
    print(f"loading profile output {profile_file} ...")

    with open(profile_file, "r") as opened_file:
        sess_time = json.load(opened_file)

    assert isinstance(sess_time, list)
    return sess_time


def parse_profile_results(sess_time, kernel_time_only=False, threshold=0):
    node_time = {}
    node_provider = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "dur" in item and "args" in item and "op_name" in item["args"]:
            if "provider" in item["args"]:
                device = "CPU" if item["args"]["provider"] == "CPUExecutionProvider" else "CUDA"
                if item["name"] not in node_provider:
                    node_provider[item["name"]] = device
                else:
                    assert node_provider[item["name"]] == device
            elif kernel_time_only:
                continue

            op_name = item["args"]["op_name"]
            if op_name in NODES_TYPE_CONTAINING_SUBGRAPH:
                continue

            if item["name"] in node_time:
                node_time[item["name"]] += item["dur"]
            else:
                node_time[item["name"]] = item["dur"]
            total += item["dur"]

    lines = []
    if (threshold > 0):
        lines.append(f"Threshold of Percentage > {threshold:.2f}%")

    lines.append("Duration\tPercentage\tProvider\tName")
    for k, v in sorted(node_time.items(), key=lambda x: x[1], reverse=True):
        provider = node_provider[k] if k in node_provider else ""
        ratio = v / total
        if ratio > threshold:
            lines.append(f"{v}\t{ratio * 100.0:5.2f}\t{provider}\t{k}")

    return lines


def group_profile_results(sess_time, kernel_time_only=False, threshold=0):
    op_time = {}
    op_records = {}
    op_cpu_time = {}
    op_cpu_records = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "dur" in item and "args" in item and "op_name" in item["args"]:
            if kernel_time_only and "provider" not in item["args"]:
                continue

            op_name = item["args"]["op_name"]

            if op_name in NODES_TYPE_CONTAINING_SUBGRAPH:
                continue

            if op_name in op_time:
                op_time[op_name] += item["dur"]
                op_records[op_name] += 1
            else:
                op_time[op_name] = item["dur"]
                op_records[op_name] = 1

            total += item["dur"]

            is_cpu = "provider" in item["args"] and item["args"]["provider"] == "CPUExecutionProvider"
            if is_cpu:
                if op_name in op_cpu_time:
                    op_cpu_time[op_name] += item["dur"]
                    op_cpu_records[op_name] += 1
                else:
                    op_cpu_time[op_name] = item["dur"]
                    op_cpu_records[op_name] = 1

    lines = ["Duration\tPercentage\tCalls\tCpu_Duration\tCpu_Calls\tName"]
    for k, v in sorted(op_time.items(), key=lambda x: x[1], reverse=True):
        calls = op_records[k]
        cpu_time = op_cpu_time[k] if k in op_cpu_time else 0
        cpu_calls = op_cpu_records[k] if k in op_cpu_records else 0
        ratio = v / total
        if ratio > threshold:
            lines.append(f"{v}\t{ratio * 100.0:5.2f}\t{calls}\t{cpu_time}\t{cpu_calls}\t{k}")
    return lines


def get_dim_from_type_proto(dim):
    return getattr(dim, dim.WhichOneof('value')) if type(dim.WhichOneof('value')) == str else None


def get_shape_from_type_proto(type_proto):
    return [get_dim_from_type_proto(d) for d in type_proto.tensor_type.shape.dim]


def create_dummy_inputs(onnx_model_path, batch_size, sequence_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    dummy_inputs = {}
    for graph_input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(graph_input.type)
        symbol_dims = []
        for i, dim in enumerate(shape):
            if isinstance(dim, str):
                symbol_dims.append(i)

        # allowed symbolic dimensions: batch_size and sequence_length
        if len(symbol_dims) > 2:
            return None
        if len(symbol_dims) > 0:
            shape[symbol_dims[0]] = batch_size
        if len(symbol_dims) > 1:
            shape[symbol_dims[1]] = sequence_length

        elem_type = graph_input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)
        data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[graph_input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs


def create_gpt2_inputs(onnx_model_path, batch_size, sequence_length, past_sequence_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    # The symbolic name shall be same as those used in Gpt2Helper.export_onnx(...) function.
    symbols = {
        'batch_size': batch_size,
        'seq_len': sequence_length,
        'past_seq_len': past_sequence_length,
        'total_seq_len': sequence_length + past_sequence_length
    }

    dummy_inputs = {}
    for graph_input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(graph_input.type)
        for i, dim in enumerate(shape):
            if isinstance(dim, str) and dim not in symbols.keys():
                raise RuntimeError(f"symbol is not supported: {dim}")
            else:
                shape[i] = symbols[dim]

        elem_type = graph_input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)
        data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[graph_input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs


def create_longformer_inputs(onnx_model_path, batch_size, sequence_length, global_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    symbols = {'batch_size': batch_size, 'sequence_length': sequence_length}

    dummy_inputs = {}
    for graph_input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(graph_input.type)
        for i, dim in enumerate(shape):
            if isinstance(dim, str) and dim not in symbols.keys():
                raise RuntimeError(f"symbol is not supported: {dim}")
            else:
                shape[i] = symbols[dim]

        elem_type = graph_input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)

        if "global" in graph_input.name:
            data = numpy.zeros(shape, dtype=data_type)
            data[:, :global_length] = 1
        else:
            data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[graph_input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs


def run(args):
    num_threads = args.thread_num if args.thread_num > 0 else psutil.cpu_count(logical=False)

    # Set OMP environment variable before importing onnxruntime. Needed for cpu only, and no impact for onnxruntime-gpu package.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)

    all_inputs = None
    if args.dummy_inputs == 'bert':
        all_inputs = create_bert_inputs(args.model, args.batch_size, args.sequence_length, args.samples,
                                        args.input_ids_name, args.segment_ids_name, args.input_mask_name)
    elif args.dummy_inputs == 'gpt2':
        all_inputs = create_gpt2_inputs(args.model, args.batch_size, args.sequence_length, args.past_sequence_length,
                                        args.samples)
    elif args.dummy_inputs == 'longformer':
        all_inputs = create_longformer_inputs(args.model, args.batch_size, args.sequence_length, args.global_length,
                                              args.samples)
    else:  # default
        all_inputs = create_dummy_inputs(args.model, args.batch_size, args.sequence_length, args.samples)

    profile_file = run_profile(args.model, args.use_gpu, args.basic_optimization, args.thread_num, all_inputs)

    profile_records = load_profile_json(profile_file)

    lines = parse_profile_results(profile_records, args.kernel_time_only, args.threshold)

    lines.append("-" * 64)
    lines += group_profile_results(profile_records, args.kernel_time_only, args.threshold)

    return lines


if __name__ == '__main__':
    arguments = parse_arguments()
    print("Arguments", arguments)

    from benchmark_helper import setup_logger
    setup_logger(arguments.verbose)

    results = run(arguments)

    print("Results:")
    print("-" * 64)
    for line in results:
        print(line)
