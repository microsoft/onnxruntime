import os
import argparse
import json
import psutil
import numpy
from onnx import TensorProto
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
        default=0.01,
        help="Threshold of run time ratio among all nodes. Nodes with larger ratio will show in top expensive nodes.")

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="number of threads to use")

    parser.add_argument('--input_ids_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for input IDs, for bert")
    parser.add_argument('--segment_ids_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for segment IDs, for bert")
    parser.add_argument('--input_mask_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for attention mask, for bert")

    parser.add_argument('--dummy_inputs',
                        required=False,
                        default='default',
                        choices=['bert', 'gpt2', 'longformer', 'default'],
                        help="Type of model inputs. The default will create dummy inputs with ones.")

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
    """Parse profile data and output nodes in two sections - nodes in the original order, and top expensive nodes.

    Args:
        sess_time (List[Dict]): profile data
        kernel_time_only (bool, optional): Only include items for kernel time. Defaults to False.
        threshold (int, optional): Minimum ratio of duration among all. Defaults to 0.

    Returns:
        List[str]: lines of string for output.
    """
    node_name_list = []
    node_time = {}
    node_freq = {}
    node_provider = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "dur" in item and "args" in item and "op_name" in item["args"]:
            node_name = item["name"].replace("_kernel_time", "").replace("_fence_before",
                                                                         "").replace("_fence_after", "")

            if "provider" in item["args"]:
                device = "CPU" if item["args"]["provider"] == "CPUExecutionProvider" else "CUDA"
                if node_name not in node_provider:
                    node_provider[node_name] = device
                else:
                    assert node_provider[node_name] == device
            elif kernel_time_only:
                continue

            op_name = item["args"]["op_name"]
            if op_name in NODES_TYPE_CONTAINING_SUBGRAPH:
                continue

            if node_name in node_time:
                node_time[node_name] += item["dur"]
                node_freq[node_name] += 1
            else:
                node_time[node_name] = item["dur"]
                node_freq[node_name] = 1
                node_name_list.append(node_name)

            total += item["dur"]

    # Output items in the original order.
    lines = [
        "Results:", "-" * 64,
        "Duration(μs)\tPercentage\tBefore(Exclusive)\tAfter(Inclusive)\tCalls\tProvider\tNode_Name"
    ]
    before_percentage = 0.0
    for node_name in node_name_list:
        duration = node_time[node_name]
        calls = node_freq[node_name]
        avg_time = duration / float(calls)
        percentage = (duration / total) * 100.0
        provider = node_provider[node_name] if node_name in node_provider else ""
        lines.append(
            f"{avg_time:.1f}\t{percentage:5.2f}\t{before_percentage:5.1f}\t{100.0 - before_percentage:5.1f}\t{calls}\t{provider}\t{node_name}"
        )
        before_percentage += percentage

    # Output items with run time ratio > thresholds, and sorted by duration in the descending order.
    lines.append(f"\nTop expensive nodes with threshold={threshold:.2f}:")
    lines.append("-" * 64)
    lines.append("Duration(μs)\tPercentage\tProvider\tName")
    for node_name, duration in sorted(node_time.items(), key=lambda x: x[1], reverse=True):
        ratio = duration / total
        if ratio < threshold:
            continue

        calls = node_freq[node_name]
        avg_time = duration / float(calls)
        provider = node_provider[node_name] if node_name in node_provider else ""
        lines.append(f"{avg_time:.1f}\t{ratio * 100.0:5.2f}\t{provider}\t{node_name}")

    return lines


def group_profile_results(sess_time, kernel_time_only, use_gpu):
    """Group results by operator name.

    Args:
        sess_time (List[Dict]): profile data
        kernel_time_only (bool): Only include items for kernel time.
        use_gpu (bool): GPU is used in profiling or not.

    Returns:
        List[str]: lines of string for output.
    """
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

    if use_gpu:
        lines = ["Average(μs)\tTotal(μs)\tTotal_Percentage\tCalls\tCpu_Duration\tCpu_Calls\tName"]
    else:
        lines = ["Average(μs)\tTotal(μs)\tTotal_Percentage\tCalls\tName"]

    for op_name, duration in sorted(op_time.items(), key=lambda x: x[1], reverse=True):
        ratio = duration / total
        calls = op_records[op_name]
        cpu_time = op_cpu_time[op_name] if op_name in op_cpu_time else 0
        cpu_calls = op_cpu_records[op_name] if op_name in op_cpu_records else 0
        avg_time = duration / float(calls)

        if use_gpu:
            lines.append(
                f"{avg_time:.1f}\t{duration}\t{ratio * 100.0:5.2f}\t{calls}\t{cpu_time}\t{cpu_calls}\t{op_name}")
        else:
            lines.append(f"{avg_time:.1f}\t{duration}\t{ratio * 100.0:5.2f}\t{calls}\t{op_name}")

    return lines


def get_dim_from_type_proto(dim):
    return getattr(dim, dim.WhichOneof('value')) if type(dim.WhichOneof('value')) == str else None


def get_shape_from_type_proto(type_proto):
    return [get_dim_from_type_proto(d) for d in type_proto.tensor_type.shape.dim]


def create_dummy_inputs(onnx_model, batch_size, sequence_length, samples):
    """Create dummy inputs for ONNX model.

    Args:
        onnx_model (OnnxModel): ONNX model
        batch_size (int): batch size
        sequence_length (int): sequence length
        samples (int): number of samples

    Returns:
        List[Dict]: list of inputs
    """
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


def create_bert_inputs(onnx_model,
                       batch_size,
                       sequence_length,
                       samples,
                       input_ids_name=None,
                       segment_ids_name=None,
                       input_mask_name=None):
    """Create dummy inputs for BERT model.

    Args:
        onnx_model (OnnxModel): ONNX model
        batch_size (int): batch size
        sequence_length (int): sequence length
        samples (int): number of samples
        input_ids_name (str, optional): Name of graph input for input IDs. Defaults to None.
        segment_ids_name (str, optional): Name of graph input for segment IDs. Defaults to None.
        input_mask_name (str, optional): Name of graph input for attention mask. Defaults to None.

    Returns:
        List[Dict]: list of inputs
    """
    from bert_test_data import find_bert_inputs, generate_test_data
    input_ids, segment_ids, input_mask = find_bert_inputs(onnx_model, input_ids_name, segment_ids_name, input_mask_name)
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


def create_gpt2_inputs(onnx_model, batch_size, sequence_length, past_sequence_length, samples):
    """Create dummy inputs for GPT-2 model.

    Args:
        onnx_model (OnnxModel): ONNX model
        batch_size (int): batch size
        sequence_length (int): sequence length
        past_sequence_length (int): past sequence length
        samples (int): number of samples

    Raises:
        RuntimeError: symbolic is not supported. Use the tool convert_to_onnx.py to export ONNX model instead.

    Returns:
        List[Dict]: list of inputs
    """
    # The symbolic names shall be same as those used in Gpt2Helper.export_onnx(...) function.
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


def create_longformer_inputs(onnx_model, batch_size, sequence_length, global_length, samples):
    """Create dummy inputs for Longformer model.

    Args:
        onnx_model (OnnxModel): ONNX model
        batch_size (int): batch size
        sequence_length (int): sequence length
        global_length (int): number of global tokens
        samples (int): number of samples

    Raises:
        RuntimeError: symbolic is not supported. Use the tool convert_longformer_to_onnx.py to export ONNX model instead.

    Returns:
        List[Dict]: list of inputs
    """
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

    from onnx import load
    from onnx_model import OnnxModel
    onnx_model = OnnxModel(load(args.model))

    all_inputs = None
    if args.dummy_inputs == 'bert':
        all_inputs = create_bert_inputs(onnx_model, args.batch_size, args.sequence_length, args.samples,
                                        args.input_ids_name, args.segment_ids_name, args.input_mask_name)
    elif args.dummy_inputs == 'gpt2':
        all_inputs = create_gpt2_inputs(onnx_model, args.batch_size, args.sequence_length, args.past_sequence_length,
                                        args.samples)
    elif args.dummy_inputs == 'longformer':
        all_inputs = create_longformer_inputs(onnx_model, args.batch_size, args.sequence_length, args.global_length,
                                              args.samples)
    else:  # default
        all_inputs = create_dummy_inputs(onnx_model, args.batch_size, args.sequence_length, args.samples)

    profile_file = run_profile(args.model, args.use_gpu, args.basic_optimization, args.thread_num, all_inputs)

    profile_records = load_profile_json(profile_file)

    lines = parse_profile_results(profile_records, args.kernel_time_only, args.threshold)

    lines.append("\nGrouped by operator type:")
    lines.append("-" * 64)
    lines += group_profile_results(profile_records, args.kernel_time_only, args.use_gpu)

    return lines


if __name__ == '__main__':
    arguments = parse_arguments()
    print("Arguments", arguments)

    from benchmark_helper import setup_logger
    setup_logger(arguments.verbose)

    results = run(arguments)
    for line in results:
        print(line)
