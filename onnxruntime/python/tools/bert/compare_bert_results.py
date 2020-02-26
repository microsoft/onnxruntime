#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# It is a tool to compare the inference results of the original model and optimized model.

import sys
import argparse
import numpy as np
import os
import onnxruntime
import random
from pathlib import Path
import statistics
import onnx
import onnx.utils
import psutil
import csv
import timeit
from datetime import datetime
from onnx import ModelProto, TensorProto, numpy_helper
from OnnxModel import OnnxModel
from bert_model_optimization import optimize_by_onnxruntime

def get_graph_input_from_embed_node(onnx_model, embed_node, input_index):
    assert input_index < len(embed_node.input)

    input = embed_node.input[input_index]
    graph_input = onnx_model.find_graph_input(input)
    if graph_input is None:
        parent_node = onnx_model.get_parent(embed_node, input_index)
        if parent_node is not None and parent_node.op_type == 'Cast':
            graph_input = onnx_model.find_graph_input(parent_node.input[0])
    return graph_input

def fake_input_ids_data(input_ids, batch_size, sequence_length, dictionary_size):
    """
    Fake data based on the graph input of input ids.
    Args:
        input_ids (TensorProto): graph input of input tensor.
    Returns:
        data (np.array): the data for input tensor
    """
    assert input_ids.type.tensor_type.elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
    
    data = np.random.randint(dictionary_size, size=(batch_size, sequence_length), dtype=np.int32)

    if input_ids.type.tensor_type.elem_type == TensorProto.FLOAT:
        data = np.float32(data)
    elif input_ids.type.tensor_type.elem_type == TensorProto.INT64:
        data = np.int64(data)

    return data

def fake_segment_ids_data(segment_ids, batch_size, sequence_length):
    """
    Fake data based on the graph input of segment_ids.
    Args:
        segment_ids (TensorProto): graph input of input tensor.
    Returns:
        data (np.array): the data for input tensor
    """
    assert segment_ids.type.tensor_type.elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
    
    data = np.zeros((batch_size, sequence_length), dtype=np.int32)

    if segment_ids.type.tensor_type.elem_type == TensorProto.FLOAT:
        data = np.float32(data)
    elif segment_ids.type.tensor_type.elem_type == TensorProto.INT64:
        data = np.int64(data)


    return data

def fake_input_mask_data(input_mask, batch_size, sequence_length, random_mask_length):
    """
    Fake data based on the graph input of segment_ids.
    Args:
        segment_ids (TensorProto): graph input of input tensor.
    Returns:
        data (np.array): the data for input tensor
    """
    assert input_mask.type.tensor_type.elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]

    if random_mask_length:
        actual_seq_len = random.randint(int(sequence_length * 2 / 3), sequence_length)
        data = np.zeros((batch_size, sequence_length), dtype=np.int32)
        temp = np.ones((batch_size, actual_seq_len), dtype=np.int32)
        data[:temp.shape[0],:temp.shape[1]]=temp
    else:
        data = np.ones((batch_size, sequence_length), dtype=np.int32)

    if input_mask.type.tensor_type.elem_type == TensorProto.FLOAT:
        data = np.float32(data)
    elif input_mask.type.tensor_type.elem_type == TensorProto.INT64:
        data = np.int64(data)

    return data

def output_test_data(output_path, test_case_id, inputs, result, output_names):
    """
    Output test data so that we can use onnxruntime_perf_test.exe to check performance laster.
    """
    path = os.path.join(output_path, 'test_data_set_' + str(test_case_id))
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    index = 0
    for name, data in inputs.items():
        tensor = numpy_helper.from_array(data, name)
        with open(os.path.join(path, 'input_{}.pb'.format(index)), 'wb') as f:
            f.write(tensor.SerializeToString())
        index += 1

def fake_test_data(batch_size, sequence_length, test_cases, dictionary_size, verbose, random_seed, input_ids, segment_ids, input_mask, random_mask_length):
    """
    Generate fake input data for test.
    """
    np.random.seed(random_seed)

    all_inputs = []
    for test_case in range(test_cases):
        input_1 = fake_input_ids_data(input_ids, batch_size, sequence_length, dictionary_size)
        input_2 = fake_segment_ids_data(segment_ids, batch_size, sequence_length)
        input_3 = fake_input_mask_data(input_mask, batch_size, sequence_length, random_mask_length)
        inputs = {input_ids.name: input_1,
                  segment_ids.name: input_2,
                  input_mask.name: input_3
                 }
        if verbose and len(all_inputs) == 0:
            print("Example inputs", inputs)
        all_inputs.append(inputs)
    return all_inputs

def get_bert_inputs(onnx_file):
    """
    Get graph inputs for bert model.
    First, we will deduce from EmbedLayerNormalization node. If not found, we will guess based on naming.
    """
    model = ModelProto()
    with open(onnx_file, "rb") as f:
        model.ParseFromString(f.read())

    onnx_model = OnnxModel(model)

    graph_inputs = onnx_model.get_graph_inputs_excluding_initializers()
    if len(graph_inputs) != 3:
        raise ValueError("Expect the graph to have 3 inputs. Got {}".format(len(graph_inputs)))

    embed_nodes = onnx_model.get_nodes_by_op_type('EmbedLayerNormalization')
    if len(embed_nodes) == 1:
        embed_node = embed_nodes[0]
        input_ids = get_graph_input_from_embed_node(onnx_model, embed_node, 0)
        segment_ids = get_graph_input_from_embed_node(onnx_model, embed_node, 1)
        input_mask = get_graph_input_from_embed_node(onnx_model, embed_node, 7)
        return input_ids, segment_ids, input_mask

    # Try guess the inputs based on naming.
    input_ids = None
    segment_ids = None
    input_mask = None
    for input in graph_inputs:
        input_name_lower = input.name.lower()
        if "mask" in input_name_lower: # matches input with name like "attention_mask" or "input_mask"
            input_mask = input
        elif "token" in input_name_lower or "segment" in input_name_lower: # matches input with name like "segment_ids" or "token_type_ids"
            segment_ids = input
        else:
            input_ids = input

    if input_ids and segment_ids and input_mask:
        return input_ids, segment_ids, input_mask

    raise ValueError("Fail to assign 3 inputs. You might try rename the graph inputs.")

def onnxruntime_inference(session, all_inputs, output_names):
    results = []
    latency_list = []
    for test_case_id, inputs in enumerate(all_inputs):
        start_time = timeit.default_timer()
        result = session.run(output_names, inputs)
        latency = timeit.default_timer() - start_time
        results.append(result)
        latency_list.append(latency)
    return results, latency_list

def generate_test_data(batch_size, sequence_length, test_cases, seed, verbose, input_ids, segment_ids, input_mask, random_mask_length):
    dictionary_size = 10000
    all_inputs = fake_test_data(batch_size, sequence_length, test_cases, dictionary_size, verbose, seed, input_ids, segment_ids, input_mask, random_mask_length)
    if len(all_inputs) != test_cases:
        print("Failed to create test data for test.")
    return all_inputs

def create_session(model_path, use_gpu, use_openmp, graph_optimization_level, num_threads, wait_policy):
    execution_providers = ['CPUExecutionProvider'] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = onnxruntime.SessionOptions()
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = graph_optimization_level
    if not use_openmp:
        sess_options.intra_op_num_threads=num_threads
        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]
        if "OMP_WAIT_POLICY" in os.environ:
            del os.environ["OMP_WAIT_POLICY"]
    else:
        sess_options.intra_op_num_threads=1
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["OMP_WAIT_POLICY"] = wait_policy

    session = onnxruntime.InferenceSession(model_path, sess_options, providers=execution_providers)
    if use_gpu:
        assert 'CUDAExecutionProvider' in session.get_providers()
    return session

def run_model(baseline_model, all_inputs, use_gpu, use_openmp, graph_optimization_level):
    session = create_session(baseline_model, use_gpu, use_openmp, graph_optimization_level, num_threads=psutil.cpu_count(logical=True), wait_policy='ACTIVE')
    output_names = [output.name for output in session.get_outputs()]
    results, latency_list = onnxruntime_inference(session, all_inputs, output_names)
    return results, latency_list, output_names

def compare(baseline_results, treatment_results, verbose, rtol=1e-3, atol=1e-4):
    # Validate the output of baseline and treatment, to make sure the results are similar.
    diff_count = 0
    first_diff = True
    max_rel_diff = []
    max_abs_diff = []
    for test_case_id, results in enumerate(baseline_results):
        treatment_first_output = treatment_results[test_case_id][0].tolist()
        if not np.allclose(results[0].tolist(), treatment_first_output, rtol=rtol, atol=atol):
            diff_count += 1
            if verbose and first_diff:
                print("baseline={}\ntreatment={}".format(results[0].tolist(), treatment_first_output))
                first_diff = False
        max_rel_diff.append(np.amax(np.abs((treatment_results[test_case_id][0] - results[0]) / results[0])))
        max_abs_diff.append(np.amax(np.abs(treatment_results[test_case_id][0] - results[0])))

    print("{} out of {} results are not close (rtol={}, atol={}).".format(diff_count, len(baseline_results), rtol, atol))

    max_abs_diff_value = max(max_abs_diff)
    print("maximum absolute difference={} in test case {}".format(max_abs_diff_value, max_abs_diff.index(max_abs_diff_value)))

    max_rel_diff_value = max(max_rel_diff)
    print("maximum relative difference={} in test case {}".format(max_rel_diff_value, max_rel_diff.index(max_rel_diff_value)))

def run_test(baseline_model, optimized_model, output_dir, batch_size, sequence_length, use_gpu, test_cases, seed, use_openmp, verbose, rtol, atol):
    # Try deduce input names from optimized model.
    input_ids, segment_ids, input_mask = get_bert_inputs(optimized_model)

    # Use random mask length for accuracy test. It might introduce slight inflation in latency reported in this script.
    all_inputs = generate_test_data(batch_size, sequence_length, test_cases, seed, verbose, input_ids, segment_ids, input_mask, random_mask_length=True)

    baseline_results, baseline_latency, output_names = run_model(baseline_model, all_inputs, use_gpu, use_openmp, onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)
    print("baseline average latency: {} ms".format(statistics.mean(baseline_latency) * 1000))

    for i, inputs in enumerate(all_inputs):
        output_test_data(output_dir, i, inputs, baseline_results[i], output_names)

    treatment_results, treatment_latency, treatment_output_names = run_model(optimized_model, all_inputs, use_gpu, use_openmp, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
    print("treatment average latency: {} ms".format(statistics.mean(treatment_latency) * 1000))

    # Validate the output of baseline and treatment, to make sure the results are similar.
    compare(baseline_results, treatment_results, verbose, rtol, atol)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_model', required=True, type=str,
                        help="baseline onnx model path")

    parser.add_argument('--optimized_model', required=False, type=str, default=None,
                        help="optimized model for the baseline model. They shall have same inputs. If it is None, an optimized model will be generated using OnnxRuntime.")

    parser.add_argument('--output_dir', required=False, type=str, default=None,
                        help="output test data path. If not specified, we create a sub-directory under the directory of the optimized model.")

    parser.add_argument('--batch_size', required=True, type=int,
                        help="batch size of input")

    parser.add_argument('--sequence_length',  required=True, type=int,
                        help="maximum sequence length of input")

    parser.add_argument('--rtol',  required=False, type=float, default=1e-3,
                        help="relative tolerance")

    parser.add_argument('--atol',  required=False, type=float, default=1e-4,
                        help="absolute tolerance")

    parser.add_argument('--samples',  required=False, type=int, default=100,
                        help="number of test cases to be generated")

    parser.add_argument('--seed',  required=False, type=int, default=3,
                        help="random seed")

    parser.add_argument('--use_gpu',  required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--no_openmp', required=False, action='store_true', help="do not use openmp")
    parser.set_defaults(no_openmp=False)

    parser.add_argument('--verbose', required=False, action='store_true', help="print verbose information")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    optimized_model = optimize_by_onnxruntime(args.baseline_model, args.use_gpu) if (args.optimized_model is None) else args.optimized_model

    if args.use_gpu and ('CUDAExecutionProvider' not in onnxruntime.get_available_providers()):
        print("Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu.")

    output_dir= args.output_dir
    if output_dir is None:
        # Default output directory is under the same directory of optimized model.
        p = Path(optimized_model)
        output_dir = os.path.join(p.parent, "batch_{}_seq_{}".format(args.batch_size, args.sequence_length))

    # create the output directory if not existed
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    run_test(
        args.baseline_model,
        optimized_model,
        output_dir,
        args.batch_size,
        args.sequence_length,
        args.use_gpu,
        args.samples,
        args.seed,
        not args.no_openmp,
        args.verbose,
        args.rtol,
        args.atol)

if __name__ == "__main__":
    main()
