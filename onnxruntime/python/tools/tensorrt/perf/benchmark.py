# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import copy
import json
import logging
import os
import pprint
import re
import subprocess
import sys
import tempfile
import timeit
from datetime import datetime

import coloredlogs
import numpy as np
from perf_utils import (
    acl,
    acl_ep,
    basic,
    cpu,
    cpu_ep,
    cuda,
    cuda_ep,
    cuda_fp16,
    disable,
    enable_all,
    extended,
    find_files,
    get_output,
    get_profile_metrics,
    is_standalone,
    pretty_print,
    second,
    split_and_sort_output,
    standalone_trt,
    standalone_trt_fp16,
    trt,
    trt_ep,
    trt_fp16,
)

import onnxruntime  # isort:skip
import onnx  # isort:skip
from onnx import numpy_helper  # isort:skip
import pandas as pd  # isort:skip
from write_csv import output_details, output_fail, output_latency, output_metrics  # isort:skip

debug = False
sys.path.append(".")
logger = logging.getLogger("")

ep_to_provider_list = {
    cpu: [cpu_ep],
    acl: [acl_ep],
    cuda: [cuda_ep],
    cuda_fp16: [cuda_ep],
    trt: [trt_ep, cuda_ep],
    trt_fp16: [trt_ep, cuda_ep],
}

# latency gain headers
trt_cuda_gain = "TRT_CUDA_gain(%)"
trt_cuda_fp16_gain = "TRT_CUDA_fp16_gain(%)"
trt_native_gain = "TRT_Standalone_gain(%)"
trt_native_fp16_gain = "TRT_Standalone_fp16_gain(%)"

# metadata
FAIL_MODEL_FILE = ".fail_model_map"
LATENCY_FILE = ".latency_map"
OP_METRICS_FILE = ".op_metrics_map"
SESSION_FILE = ".session_map"
MEMORY_FILE = "./temp_memory.csv"

TRT_ENGINE_CACHE_DIR_NAME = "engine_cache"


def is_dynamic(model):
    inp = model.graph.input[0]
    for dim in inp.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return True
    return False


def get_model_inputs(model):
    all_inputs = [node.name for node in model.graph.input]
    input_initializers = [node.name for node in model.graph.initializer]
    inputs = list(set(all_inputs) - set(input_initializers))
    return inputs


def run_trt_standalone(trtexec, model_name, model_path, test_data_dir, all_inputs_shape, fp16, track_memory):
    logger.debug("running standalone trt")
    onnx_model_path = "--onnx=" + model_path

    # load inputs
    input_shape = []
    loaded_inputs = []

    model = onnx.load(model_path)
    ort_inputs = get_model_inputs(model)

    output = get_output(["find", "-L", test_data_dir, "-name", "test_data*", "-type", "d"])
    test_data_dir_0 = split_and_sort_output(output)[0]

    for i in range(len(ort_inputs)):
        name = ort_inputs[i]
        loaded_input = name + ":" + test_data_dir_0 + "/" + str(i) + ".bin"
        logger.debug(loaded_input)
        shape = []
        for j in all_inputs_shape[i]:
            shape.append(str(j))
        shape = "x".join(shape)
        shape = name + ":" + shape
        input_shape.append(shape)
        loaded_inputs.append(loaded_input)

    shapes_arg = "--optShapes=" + ",".join(input_shape)
    inputs_arg = "--loadInputs=" + ",".join(loaded_inputs)
    result = {}
    command = [
        trtexec,
        onnx_model_path,
        "--duration=50",
        "--percentile=90",
        "--workspace=4096",
    ]
    command.extend([inputs_arg])

    # add benchmarking flags
    if is_dynamic(model):
        command.extend([shapes_arg])
    if fp16:
        command.extend(["--fp16"])

    # save engine
    engine_suffix = "_trtexec_fp16.engine" if fp16 else "_trtexec.engine"
    engine_name = model_name + engine_suffix
    save_command = command + ["--saveEngine=" + engine_name]
    logger.debug(save_command)
    out = get_output(save_command)

    # load engine and inference
    load_command = command + ["--loadEngine=" + engine_name]
    logger.debug(load_command)

    mem_usage = None
    p = None
    success = False
    if track_memory:
        p = start_memory_tracking()
        try:
            out = get_output(load_command)
            success = True
            mem_usage = end_memory_tracking(p, success)
        except Exception as excpt:
            end_memory_tracking(p, success)
            raise (excpt)
    else:
        out = get_output(load_command)

    # parse trtexec output
    tmp = out.split("\n")
    target_list = []
    for t in tmp:
        if "mean" in t:
            target_list.append(t)

        if "percentile" in t:
            target_list.append(t)

    target = target_list[2]
    avg_latency_match = re.search("mean = (.*?) ms", target)
    if avg_latency_match:
        result["average_latency_ms"] = avg_latency_match.group(1)  # extract number
    percentile_match = re.search("percentile\(90%\) = (.*?) ms", target)
    if percentile_match:
        result["latency_90_percentile"] = percentile_match.group(1)  # extract number
    if mem_usage:
        result["memory"] = mem_usage

    logger.debug(result)

    return result


def get_latency_result(runtimes, batch_size):
    latency_ms = sum(runtimes) / float(len(runtimes)) * 1000.0
    latency_variance = np.var(runtimes, dtype=np.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    result = {
        "test_times": len(runtimes),
        "latency_variance": "{:.2f}".format(latency_variance),
        "latency_90_percentile": "{:.2f}".format(np.percentile(runtimes, 90) * 1000.0),
        "latency_95_percentile": "{:.2f}".format(np.percentile(runtimes, 95) * 1000.0),
        "latency_99_percentile": "{:.2f}".format(np.percentile(runtimes, 99) * 1000.0),
        "average_latency_ms": "{:.2f}".format(latency_ms),
        "QPS": "{:.2f}".format(throughput),
    }
    return result


def get_ort_session_inputs_and_outputs(name, session, ort_input):

    sess_inputs = {}
    sess_outputs = None

    if "bert_squad" in name.lower() or "bert-squad" in name.lower():
        unique_ids_raw_output = ort_input[0]
        input_ids = ort_input[1]
        input_mask = ort_input[2]
        segment_ids = ort_input[3]

        sess_inputs = {
            "unique_ids_raw_output___9:0": unique_ids_raw_output,
            "input_ids:0": input_ids[0:1],
            "input_mask:0": input_mask[0:1],
            "segment_ids:0": segment_ids[0:1],
        }
        sess_outputs = ["unique_ids:0", "unstack:0", "unstack:1"]

    elif "bidaf" in name.lower():
        sess_inputs = {
            "context_word": ort_input[0],
            "context_char": ort_input[2],
            "query_word": ort_input[1],
            "query_char": ort_input[3],
        }
        sess_outputs = ["start_pos", "end_pos"]

    elif "yolov4" in name.lower():
        sess_inputs[session.get_inputs()[0].name] = ort_input[0]
        sess_outputs = ["Identity:0"]

    else:
        sess_inputs = {}
        sess_outputs = []
        for i in range(len(session.get_inputs())):
            sess_inputs[session.get_inputs()[i].name] = ort_input[i]
        for i in range(len(session.get_outputs())):
            sess_outputs.append(session.get_outputs()[i].name)
    return (sess_inputs, sess_outputs)


def get_max_memory():
    df = pd.read_csv(MEMORY_FILE)
    pid = df["pid"].iloc[0]
    mem_series = df.loc[df["pid"] == pid, " used_gpu_memory [MiB]"]
    max_mem = max(mem_series.str.replace(" MiB", "").astype(int))
    return max_mem


def start_memory_tracking():
    logger.debug("starting memory tracking process")
    p = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv",
            "-l",
            "1",
            "-f",
            MEMORY_FILE,
        ]
    )
    return p


def end_memory_tracking(p, success):
    logger.debug("terminating memory tracking process")
    p.terminate()
    p.wait()
    p.kill()
    mem_usage = None
    if success:
        mem_usage = get_max_memory()
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    return mem_usage


def inference_ort_with_ep(exec_provider, session, repeat_times, sess_outputs, sess_inputs, with_binding, io_binding):
    if with_binding and cpu not in exec_provider:  # other eps utilize python binding
        runtime = timeit.repeat(
            lambda: session.run_with_iobinding(io_binding),
            number=1,
            repeat=repeat_times,
        )
    else:
        runtime = timeit.repeat(
            lambda: session.run(sess_outputs, sess_inputs),
            number=1,
            repeat=repeat_times,
        )
    success = True
    return runtime, success


def inference_ort(
    args,
    name,
    session,
    exec_provider,
    ort_inputs,
    result_template,
    repeat_times,
    batch_size,
    track_memory,
):
    runtimes = []
    if args.input_data == "random":
        repeat_times = 1  # warn-up run is included in ort_inputs
    else:
        repeat_times += 1  # add warn-up run

    mem_usages = []
    p = None
    mem_usage = None
    success = False

    # get and load inputs and outputs
    for ort_input in ort_inputs:
        io_binding = session.io_binding()
        sess_inputs, sess_outputs = get_ort_session_inputs_and_outputs(name, session, ort_input)
        if debug:
            logger.debug("ORT session inputs:")
            logger.debug(sess_inputs)
            logger.debug("ORT session outputs:")
            logger.debug(sess_outputs)
        if args.io_binding:
            for name, inp in sess_inputs.items():
                io_binding.bind_cpu_input(name, inp)
            for out in sess_outputs:
                io_binding.bind_output(out)

        try:
            if track_memory:
                p = start_memory_tracking()
                runtime, success = inference_ort_with_ep(
                    exec_provider,
                    session,
                    repeat_times,
                    sess_outputs,
                    sess_inputs,
                    args.io_binding,
                    io_binding,
                )
                mem_usage = end_memory_tracking(p, success)
                mem_usages.append(mem_usage)
            else:
                runtime, success = inference_ort_with_ep(
                    exec_provider,
                    session,
                    repeat_times,
                    sess_outputs,
                    sess_inputs,
                    args.io_binding,
                    io_binding,
                )
            if args.input_data == "fix":
                runtime = runtime[1:]  # remove warmup
            runtimes += runtime

        except Exception as excpt:
            logger.error(excpt)
            if track_memory:
                end_memory_tracking(p, success)
            raise (excpt)

    if len(mem_usages) > 0:
        mem_usage = max(mem_usages)

    result = {}
    result.update(result_template)
    result.update({"io_binding": True})
    latency_result = get_latency_result(runtimes, batch_size)
    result.update(latency_result)
    return result, mem_usage


def inference_ort_and_get_prediction(name, session, ort_inputs):

    ort_outputs = []
    for ort_input in ort_inputs:
        sess_inputs, sess_outputs = get_ort_session_inputs_and_outputs(name, session, ort_input)
        if debug:
            logger.debug("ORT session inputs:")
            logger.debug(sess_inputs)
            logger.debug("ORT session outputs:")
            logger.debug(sess_outputs)

        result = session.run(sess_outputs, sess_inputs)

        if debug:
            logger.debug("ORT session output results:")
            logger.debug(result)

        # handle shape of output differently
        if "bert_squad" in name.lower():
            ort_outputs.append([result])
        elif "shufflenet-v2" in name.lower() or "shufflenet_v2" in name.lower():
            ort_outputs.append(result[0])
        else:
            ort_outputs.append(result)

    return ort_outputs


#######################################################################################################################################
# The following two lists will be generated.
#
# inputs: [[test_data_0_input_0.pb, test_data_0_input_1.pb ...], [test_data_1_input_0.pb, test_data_1_input_1.pb ...] ...]
# outputs: [[test_data_0_output_0.pb, test_data_0_output_1.pb ...], [test_data_1_output_0.pb, test_data_1_output_1.pb ...] ...]
#######################################################################################################################################
def load_onnx_model_zoo_test_data(path, all_inputs_shape, fp16):
    logger.debug("Parsing test data in %s ...", path)
    output = get_output(["find", path, "-name", "test_data*", "-type", "d"])
    test_data_set_dir = split_and_sort_output(output)
    logger.debug(test_data_set_dir)

    inputs = []
    outputs = []

    shape_flag = False
    # if not empty means input shape has been parsed before.
    if len(all_inputs_shape) > 0:
        shape_flag = True

    # find test data path
    for test_data_dir in test_data_set_dir:
        pwd = os.getcwd()
        os.chdir(test_data_dir)

        # load inputs and create bindings
        output = get_output(["find", ".", "-name", "input*"])
        input_data = split_and_sort_output(output)
        logger.debug(input_data)

        input_data_pb = []
        i = 0
        for data in input_data:
            tensor = onnx.TensorProto()
            with open(data, "rb") as file_handle:
                tensor.ParseFromString(file_handle.read())
                tensor_to_array = numpy_helper.to_array(tensor)
                if fp16 and tensor_to_array.dtype == np.dtype(np.float32):
                    tensor_to_array = tensor_to_array.astype(np.float16)
                tensor_to_array.tofile(str(i) + ".bin")
                input_data_pb.append(tensor_to_array)
                if not shape_flag:
                    all_inputs_shape.append(input_data_pb[-1].shape)
                logger.debug(all_inputs_shape[-1])
        inputs.append(input_data_pb)
        logger.debug("Loaded %d inputs successfully.", len(inputs))

        # load outputs
        output = get_output(["find", ".", "-name", "output*"])
        output_data = split_and_sort_output(output)

        if len(output_data) > 0 and output_data[0] != "":
            logger.debug(output_data)
            output_data_pb = []
            for data in output_data:
                tensor = onnx.TensorProto()
                with open(data, "rb") as file_handle:
                    tensor.ParseFromString(file_handle.read())

                    tensor_to_array = numpy_helper.to_array(tensor)

                    if fp16 and tensor_to_array.dtype == np.dtype(np.float32):
                        tensor_to_array = tensor_to_array.astype(np.float16)
                    output_data_pb.append(tensor_to_array)

                    logger.debug(np.array(output_data_pb[-1]).shape)
            outputs.append(output_data_pb)
            logger.debug("Loaded %d outputs successfully.", len(outputs))

        os.chdir(pwd)
    return inputs, outputs


def generate_onnx_model_random_input(test_times, ref_input):
    inputs = []

    for i in range(test_times):

        input_data = []
        for tensor in ref_input:
            shape = tensor.shape
            dtype = tensor.dtype
            if (
                dtype == np.int8
                or dtype == np.uint8
                or dtype == np.int16
                or dtype == np.uint16
                or dtype == np.int32
                or dtype == np.uint32
                or dtype == np.int64
                or dtype == np.uint64
            ):
                new_tensor = np.random.randint(0, np.max(tensor) + 1, shape, dtype)
            else:
                new_tensor = np.random.random_sample(shape).astype(dtype)

            if debug:
                logger.debug("original tensor:")
                logger.debug(tensor)
                logger.debug("new random tensor:")
                logger.debug(new_tensor)
                logger.debug("\n")

            input_data.append(new_tensor)
        inputs.append(input_data)

    return inputs


def percentage_in_allowed_threshold(excpt, percent_mismatch):
    percent_string = re.search(r"\(([^)]+)", str(excpt)).group(1)
    if "%" in percent_string:
        percentage_wrong = float(percent_string.replace("%", ""))
        return percentage_wrong < percent_mismatch
    else:
        return False  # error in output


def validate(all_ref_outputs, all_outputs, rtol, atol, percent_mismatch):
    if len(all_ref_outputs) == 0:
        logger.debug("No reference output provided.")
        return True, None

    logger.debug("Reference %d results.", len(all_ref_outputs))
    logger.debug("Predicted %d results.", len(all_outputs))
    logger.debug("rtol: %d, atol: %d", rtol, atol)

    for i in range(len(all_outputs)):
        ref_outputs = all_ref_outputs[i]
        outputs = all_outputs[i]

        for j in range(len(outputs)):
            ref_output = ref_outputs[j]
            output = outputs[j]

            # Compare the results with reference outputs
            for ref_o, o in zip(ref_output, output):
                # abs(desired-actual) < rtol * abs(desired) + atol
                try:
                    np.testing.assert_allclose(ref_o, o, rtol, atol)
                except Exception as excpt:
                    if percentage_in_allowed_threshold(excpt, percent_mismatch):
                        continue
                    logger.error(excpt)
                    return False, excpt

    logger.debug("ONNX Runtime outputs are similar to reference outputs!")
    return True, None


def update_op_metrics_map(model_to_op_metrics, model_name, ep_to_operator):
    """
    Updates `model_to_op_metrics` to include operator metrics for the given model.

    :param model_to_op_metrics: Dictionary that maps a model to a dictionary of operator metrics per EP.
                             This function sets the operator metrics.
    :param model_name: The name of the model.
    :param ep_to_operator: Dictionary that maps an EP to information on operator usage.
    """

    if len(ep_to_operator) == 0:
        return

    if model_name not in model_to_op_metrics:
        model_to_op_metrics[model_name] = {}

    for exec_provider, op_maps in ep_to_operator.items():
        if exec_provider not in model_to_op_metrics[model_name]:
            model_to_op_metrics[model_name][exec_provider] = {}

        model_to_op_metrics[model_name][exec_provider]["nodes"] = op_maps[0]
        model_to_op_metrics[model_name][exec_provider]["kernels"] = op_maps[1]


###################################################################################################
#
# model: {ep1: {error_type: xxx, error_message: xxx}, ep2: {error_type: xx, error_message: xx}}
#
###################################################################################################
def update_fail_model_map(model_to_fail_ep, model_name, exec_provider, e_type, excpt):

    if model_name in model_to_fail_ep and exec_provider in model_to_fail_ep[model_name]:
        return

    if model_name not in model_to_fail_ep:
        model_to_fail_ep[model_name] = {}

    new_map = {}
    new_map["error_type"] = e_type
    new_map["error_message"] = re.sub("^\n", "", str(excpt))
    model_to_fail_ep[model_name][exec_provider] = new_map

    # If TRT fails, TRT FP16 should fail as well
    if exec_provider == trt:
        ep_ = trt_fp16
        e_ = "skip benchmarking since TRT failed already."
        new_map_1 = {}
        new_map_1["error_type"] = e_type
        new_map_1["error_message"] = e_
        model_to_fail_ep[model_name][ep_] = new_map_1


def skip_ep(model_name, exec_provider, model_to_fail_ep):

    if model_name not in model_to_fail_ep:
        return False

    fail_ep_list = model_to_fail_ep[model_name]

    if exec_provider in fail_ep_list:
        logger.debug("Skip testing %s using %s since it has some issues.", model_name, exec_provider)
        return True

    return False


def read_map_from_file(map_file):
    with open(map_file, encoding="utf-8") as file_handle:
        try:
            data = json.load(file_handle)
        except Exception:
            return {}

    return data


def write_map_to_file(result, file_name):
    existed_result = {}
    if os.path.exists(file_name):
        existed_result = read_map_from_file(file_name)

    for model, ep_list in result.items():
        if model in existed_result:
            existed_result[model] = {**existed_result[model], **result[model]}
        else:
            existed_result[model] = result[model]

    with open(file_name, "w") as file:
        file.write(json.dumps(existed_result))  # use `json.loads` to do the reverse


def find_model_path(path):
    output = get_output(["find", "-L", path, "-name", "*.onnx"])
    model_path = split_and_sort_output(output)
    logger.debug(model_path)

    if model_path == [""]:
        return None

    target_model_path = []
    for m in model_path:
        if "by_trt_perf" in m or m.startswith("."):
            continue
        target_model_path.append(m)

    logger.debug(target_model_path)
    if len(target_model_path) > 1:
        logger.error("We expect to find only one model in " + path)
        raise

    return target_model_path[0]


def find_model_directory(path):
    output = get_output(
        [
            "find",
            "-L",
            path,
            "-maxdepth",
            "1",
            "-mindepth",
            "1",
            "-name",
            "*",
            "-type",
            "d",
        ]
    )
    model_dir = split_and_sort_output(output)
    if model_dir == [""]:
        return None

    return model_dir


def find_test_data_directory(path):
    output = get_output(["find", "-L", path, "-maxdepth", "1", "-name", "test_data*", "-type", "d"])
    test_data_dir = split_and_sort_output(output)
    logger.debug(test_data_dir)

    if test_data_dir == [""]:
        return None

    return test_data_dir


def parse_models_info_from_directory(path, models):

    test_data_dir = find_test_data_directory(path)

    if test_data_dir:
        model_name = os.path.split(path)[-1]
        model_name = model_name + "_" + os.path.split(os.path.split(path)[0])[-1]  # get opset version as model_name
        model_path = find_model_path(path)

        model = {}
        model["model_name"] = model_name
        model["model_path"] = model_path
        model["working_directory"] = path
        model["test_data_path"] = path

        models[model_name] = model

        logger.debug(model)
        return

    model_dir = find_model_directory(path)

    if model_dir:
        for dir in model_dir:
            parse_models_info_from_directory(os.path.join(path, dir), models)


def parse_models_info_from_file(root_dir, path, models):

    # default working directory
    root_working_directory = root_dir + "perf/"

    with open(path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)

        for row in data:

            if "root_working_directory" in row:
                root_working_directory = row["root_working_directory"]
                continue

            if "model_name" in row:
                models[row["model_name"]] = {}
            else:
                logger.error("Model name must be provided in models_info.json")
                raise

            model = models[row["model_name"]]

            if "working_directory" in row:
                if os.path.isabs(row["working_directory"]):
                    model["working_directory"] = row["working_directory"]
                else:
                    model["working_directory"] = os.path.join(root_working_directory, row["working_directory"])
            else:
                logger.error("Model path must be provided in models_info.json")
                raise

            if "model_path" in row:
                model["model_path"] = row["model_path"]
            else:
                logger.error("Model path must be provided in models_info.json")
                raise

            if "test_data_path" in row:
                model["test_data_path"] = row["test_data_path"]
            else:
                logger.error("Test data path must be provided in models_info.json")
                raise

            if "model_path_fp16" in row:
                model["model_path_fp16"] = row["model_path_fp16"]

            if "test_data_path_fp16" in row:
                model["test_data_path_fp16"] = row["test_data_path_fp16"]


def convert_model_from_float_to_float16(model_path, new_model_dir):
    from float16 import convert_float_to_float16
    from onnxmltools.utils import load_model, save_model

    new_model_path = os.path.join(new_model_dir, "new_fp16_model_by_trt_perf.onnx")

    if not os.path.exists(new_model_path):
        onnx_model = load_model(model_path)
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, new_model_path)

    return new_model_path


def get_test_data(fp16, test_data_dir, all_inputs_shape):
    inputs = []
    ref_outputs = []
    inputs, ref_outputs = load_onnx_model_zoo_test_data(test_data_dir, all_inputs_shape, fp16)
    return inputs, ref_outputs


def run_symbolic_shape_inference(model_path, new_model_path):
    import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer

    logger.debug("run symbolic shape inference")
    try:
        out = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx.load(model_path), auto_merge=True)
        onnx.save(out, new_model_path)
        return True, None
    except Exception as excpt:
        logger.error(excpt)
        return False, "Symbolic shape inference error"


def get_provider_options(providers, trt_ep_options, cuda_ep_options):
    provider_options = []

    for exec_provider in providers:
        if exec_provider == trt_ep:
            provider_options.append(trt_ep_options)
        elif exec_provider == cuda_ep:
            provider_options.append(cuda_ep_options)
        else:
            provider_options.append({})

    return provider_options


def time_and_create_session(model_path, providers, provider_options, session_options):
    start = datetime.now()
    session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
        provider_options=provider_options,
        sess_options=session_options,
    )
    end = datetime.now()
    creation_time = (end - start).total_seconds()
    return session, creation_time


def create_session(model_path, providers, provider_options, session_options):
    logger.debug(model_path)

    try:
        return time_and_create_session(model_path, providers, provider_options, session_options)
    except Exception as excpt:
        # shape inference required on model
        if "shape inference" in str(excpt):
            logger.debug("Using model from symbolic_shape_infer.py")
            new_model_path = model_path[:].replace(".onnx", "_new_by_trt_perf.onnx")
            if not os.path.exists(new_model_path):
                status = run_symbolic_shape_inference(model_path, new_model_path)
                if not status[0]:  # symbolic shape inference error
                    excpt = status[1]
                    raise Exception(excpt)
            return time_and_create_session(new_model_path, providers, provider_options, session_options)
        else:
            raise Exception(excpt)


def fill_trt_subgraphs(op_maps, dump_dir):
    """
    Parses dumped TensorRT subgraphs to augment operator usage information with the actual operators
    used within TRT subgraphs.

    :param op_maps: A tuple containing the parsed operator usage information for CPU nodes and GPU kernels.
    :param dump_dir: The name of the directory containing dumped TRT subgraphs.
    """

    # List of dumped subgraph files (See trt_dump_subgraphs EP option)
    subgraph_files = find_files(dump_dir, "*.onnx")

    # Dictionary that maps a TRT operator name (e.g., TRTKernel_graph_main_9492348588788950893_1) to a list of
    # 'info' dictionaries that need to be updated with subgraph information.
    op_infos_to_update = {}

    # Determine which 'info' dictionaries need to be updated.
    for op_map in op_maps:
        if trt_ep in op_map:
            for op_name, op_info in op_map[trt_ep].items():
                if op_name in op_infos_to_update:
                    op_infos_to_update[op_name].append(op_info)
                else:
                    op_infos_to_update[op_name] = [op_info]

    # Load dumped .onnx subraphs and update 'info' dictionaries.
    for op_name, op_infos in op_infos_to_update.items():
        subgraph_file = next((file_name for file_name in subgraph_files if op_name in file_name), None)

        if not subgraph_file:
            continue

        onnx_model = onnx.load(subgraph_file)
        subgraph_op_counts = {}

        for node in onnx_model.graph.node:
            op_type = node.op_type

            if op_type in subgraph_op_counts:
                subgraph_op_counts[op_type] += 1
            else:
                subgraph_op_counts[op_type] = 1

        for op_info in op_infos:
            op_info["subgraph"] = copy.deepcopy(subgraph_op_counts)


def run_model_on_ep(
    args,
    model_name,
    model_info,
    exec_provider,
    success_results,
    ep_to_latency,
    ep_to_op_metrics,
    model_to_fail_ep,
    ep_to_session,
    tmp_work_dir,
):
    """
    Benchmarks or validates the given model on the given EP.

    :param args: The command-line arguments to this script.
    :param model_name: The name of the model to run.
    :param model_info: A dictionary that contains paths to the model file and input data.
    :param exec_provider: The name of the EP (e.g., ORT-CUDAFp32) on which to run the model.
    :param success_results: List of successful results that is updated by this function.
    :param ep_to_latency: Dictionary that maps an EP to inference latency and memory usage results.
                          Updated by this function.
    :param ep_to_op_metrics: Dictionary that maps an EP to operator usage information. Updated by this function.
    :param model_to_fail_ep: Dictionary that tracks failing model and EP combinations. Updated by this function.
    :param ep_to_session: Dictionary that maps an EP to session creation latency results. Updated by this function.
    :param tmp_work_dir: Temporary directory in which to run the model + EP.
    """

    all_inputs_shape = []  # used for standalone trt
    model_work_dir = os.path.normpath(model_info["working_directory"])
    model_path = os.path.normpath(os.path.join(model_work_dir, model_info["model_path"]))
    test_data_dir = os.path.normpath(os.path.join(model_work_dir, model_info["test_data_path"]))

    os.chdir(tmp_work_dir)

    logger.info("Starting mode '%s' for %s on %s ...", args.running_mode, model_name, exec_provider)

    # Set environment variables for ort-trt benchmarking
    trt_ep_options = copy.deepcopy(args.trt_ep_options)
    if "ORT-TRT" in exec_provider:
        trt_ep_options["trt_fp16_enable"] = "True" if "Fp16" in exec_provider else "False"

        # Create/set a directory to store TRT engine caches.
        engine_cache_path = os.path.normpath(os.path.join(tmp_work_dir, TRT_ENGINE_CACHE_DIR_NAME))
        if not os.path.exists(engine_cache_path):
            os.makedirs(engine_cache_path)

        trt_ep_options["trt_engine_cache_path"] = engine_cache_path

    fp16 = False

    # use float16.py for cuda fp16 only
    if cuda_fp16 == exec_provider:

        # handle model
        if "model_path_fp16" in model_info:
            model_path = os.path.normpath(os.path.join(model_work_dir, model_info["model_path_fp16"]))

        else:
            try:
                model_path = convert_model_from_float_to_float16(model_path, tmp_work_dir)
                fp16 = True
            except Exception as excpt:
                logger.error(excpt)
                update_fail_model_map(model_to_fail_ep, model_name, exec_provider, "script error", excpt)
                return

        # handle test data
        if "test_data_path_fp16" in model_info:
            test_data_dir = os.path.normpath(os.path.join(model_work_dir, model_info["test_data_path_fp16"]))
            fp16 = False

    if standalone_trt_fp16 == exec_provider:
        fp16 = True

    inputs, ref_outputs = get_test_data(fp16, test_data_dir, all_inputs_shape)
    # generate random input data
    if args.input_data == "random":
        inputs = generate_onnx_model_random_input(args.test_times, inputs[0])

    #######################################
    # benchmark or validation
    #######################################
    if args.running_mode == "benchmark":

        # memory tracking variables
        mem_usage = None
        result = None

        # get standalone TensorRT perf
        if is_standalone(exec_provider) and args.trtexec:
            try:
                result = run_trt_standalone(
                    args.trtexec,
                    model_name,
                    model_path,
                    test_data_dir,
                    all_inputs_shape,
                    fp16,
                    args.track_memory,
                )
            except Exception as excpt:
                logger.error(excpt)
                update_fail_model_map(model_to_fail_ep, model_name, exec_provider, "runtime error", excpt)
                return

        # inference with onnxruntime ep
        else:
            # resolve providers to create session
            providers = ep_to_provider_list[exec_provider]
            provider_options = get_provider_options(providers, trt_ep_options, args.cuda_ep_options)
            options = onnxruntime.SessionOptions()

            enablement = args.graph_enablement
            if enablement == enable_all:
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            elif enablement == extended:
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            elif enablement == basic:
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            else:  # disable
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

            # create onnxruntime inference session
            try:
                sess, second_creation_time = create_session(model_path, providers, provider_options, options)

            except Exception as excpt:
                logger.error(excpt)
                update_fail_model_map(model_to_fail_ep, model_name, exec_provider, "runtime error", excpt)
                return

            if second_creation_time:
                ep_to_session[exec_provider + second] = second_creation_time

            logger.debug("Start to inference %s with %s ...", model_name, exec_provider)
            logger.debug(sess.get_providers())
            logger.debug(sess.get_provider_options())

            if sess:
                logger.debug("Model inputs nodes:")
                for input_meta in sess.get_inputs():
                    logger.debug(input_meta)
                logger.debug("Model outputs nodes:")
                for output_meta in sess.get_outputs():
                    logger.debug(output_meta)

            batch_size = 1
            result_template = {
                "engine": "onnxruntime",
                "version": onnxruntime.__version__,
                "device": exec_provider,
                "fp16": fp16,
                "io_binding": args.io_binding,
                "graph_optimizations": args.graph_enablement,
                "enable_cache": args.trt_ep_options.get("trt_engine_cache_enable", "False"),
                "model_name": model_name,
                "inputs": len(sess.get_inputs()),
                "batch_size": batch_size,
                "sequence_length": 1,
                "datetime": str(datetime.now()),
            }

            # run cpu fewer times
            repeat_times = 100 if exec_provider == cpu else args.test_times
            track_memory = False if exec_provider == cpu else args.track_memory

            # inference with ort
            try:
                result, mem_usage = inference_ort(
                    args,
                    model_name,
                    sess,
                    exec_provider,
                    inputs,
                    result_template,
                    repeat_times,
                    batch_size,
                    track_memory,
                )
            except Exception as excpt:
                logger.error(excpt)
                update_fail_model_map(model_to_fail_ep, model_name, exec_provider, "runtime error", excpt)
                return

        if result:

            ep_to_latency[exec_provider] = {}
            ep_to_latency[exec_provider]["average_latency_ms"] = result["average_latency_ms"]
            ep_to_latency[exec_provider]["latency_90_percentile"] = result["latency_90_percentile"]
            if "memory" in result:
                mem_usage = result["memory"]
            if mem_usage:
                ep_to_latency[exec_provider]["memory"] = mem_usage
            if not args.trtexec:  # skip standalone
                success_results.append(result)

    elif args.running_mode == "validate":
        validation_exemption = [trt_fp16]

        # enable profiling to generate profiling file for analysis
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = f"ort_profile_{model_name}_{exec_provider}"
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        if "ORT-TRT" in exec_provider:
            trt_ep_options["trt_dump_subgraphs"] = "True"

        providers = ep_to_provider_list[exec_provider]
        provider_options = get_provider_options(providers, trt_ep_options, args.cuda_ep_options)

        # create onnxruntime inference session
        try:
            sess, creation_time = create_session(model_path, providers, provider_options, options)

        except Exception as excpt:
            logger.error(excpt)
            update_fail_model_map(model_to_fail_ep, model_name, exec_provider, "runtime error", excpt)
            return

        if creation_time:
            ep_to_session[exec_provider] = creation_time

        sess.disable_fallback()

        logger.debug("Start to inference %s with %s ...", model_name, exec_provider)
        logger.debug(sess.get_providers())
        logger.debug(sess.get_provider_options())

        if sess:
            logger.debug("Model inputs nodes:")
            for input_meta in sess.get_inputs():
                logger.debug(input_meta)
            logger.debug("Model outputs nodes:")
            for output_meta in sess.get_outputs():
                logger.debug(output_meta)

        # run inference and validate the result
        #
        # currently skip TensorRT float16 validation intentionally
        if exec_provider not in validation_exemption:
            try:
                ort_outputs = inference_ort_and_get_prediction(model_name, sess, inputs)

                status = validate(
                    ref_outputs,
                    ort_outputs,
                    args.rtol,
                    args.atol,
                    args.percent_mismatch,
                )
                if not status[0]:
                    update_fail_model_map(
                        model_to_fail_ep,
                        model_name,
                        exec_provider,
                        "result accuracy issue",
                        status[1],
                    )
                    return
            except Exception as excpt:
                logger.error(excpt)
                update_fail_model_map(model_to_fail_ep, model_name, exec_provider, "runtime error", excpt)
                return

            # Run inference again. the reason is that some ep like tensorrt
            # it takes much longer time to generate graph on first run and
            # we need to skip the perf result of that expensive run.
            inference_ort_and_get_prediction(model_name, sess, inputs)
        else:
            inference_ort_and_get_prediction(model_name, sess, inputs)
            inference_ort_and_get_prediction(model_name, sess, inputs)

        sess.end_profiling()

        # get metrics from profiling file
        metrics = get_profile_metrics(tmp_work_dir, options.profile_file_prefix, logger)
        if metrics:
            if "ORT-TRT" in exec_provider:
                fill_trt_subgraphs(metrics, tmp_work_dir)

            ep_to_op_metrics[exec_provider] = metrics


def run_onnxruntime(args, models):
    """
    Benchmarks or validates the given models over the provided set of EPs.

    :param args: The command-line arguments to this script. Contains the list of EPs to use.
    :param models: Dictionary of models to run. The keys are model names and the values are dictionaries containing
                   paths to the model files and input data.

    :return: A tuple containing aggregated metrics/results.
    """

    success_results = []
    model_to_latency = {}  # model -> cuda and tensorrt latency
    model_to_op_metrics = {}  # model -> metrics from profiling file
    model_to_fail_ep = {}  # model -> failing ep
    model_to_session = {}  # models -> session creation time

    if args.running_mode == "benchmark" and os.path.exists(SESSION_FILE):
        model_to_session = read_map_from_file(SESSION_FILE)

    ep_list = []
    if args.ep:
        ep_list.append(args.ep)
    else:
        if args.fp16:
            ep_list = [cpu, cuda, trt, cuda_fp16, trt_fp16]
        else:
            ep_list = [cpu, cuda, trt]

    if os.path.exists(FAIL_MODEL_FILE):
        model_to_fail_ep = read_map_from_file(FAIL_MODEL_FILE)

    init_dir = os.getcwd()

    # Run benchmarking or validation for every model + EP combination.
    for name, model_info in models.items():
        ep_to_latency = {}
        ep_to_op_metrics = {}
        ep_to_session = {}

        for exec_provider in ep_list:

            # Skip model + EP combinations that have already failed in a previous run.
            if skip_ep(name, exec_provider, model_to_fail_ep):
                continue

            if not is_standalone(exec_provider):
                ep_ = ep_to_provider_list[exec_provider][0]
                if ep_ not in onnxruntime.get_available_providers():
                    logger.error("No {ep_} support")
                    continue

            # Create a temporary directory for this run, which may create profiles, subgraph dumps, and TRT engines.
            # The temporary directory is created in '/tmp/' and is automatically deleted after scope exit.
            with tempfile.TemporaryDirectory() as temp_dir:
                run_model_on_ep(
                    args,
                    name,
                    model_info,
                    exec_provider,
                    success_results,
                    ep_to_latency,
                    ep_to_op_metrics,
                    model_to_fail_ep,
                    ep_to_session,
                    temp_dir,
                )

        model_to_latency[name] = ep_to_latency
        model_to_session[name] = ep_to_session
        update_op_metrics_map(model_to_op_metrics, name, ep_to_op_metrics)

    os.chdir(init_dir)

    return (
        success_results,
        model_to_latency,
        model_to_fail_ep,
        model_to_op_metrics,
        model_to_session,
    )


def calculate_gain(value, ep1, ep2):
    ep1_latency = float(value[ep1]["average_latency_ms"])
    ep2_latency = float(value[ep2]["average_latency_ms"])
    gain = (ep2_latency - ep1_latency) * 100 / ep2_latency
    return gain


def add_improvement_information(model_to_latency):
    for key, value in model_to_latency.items():
        if trt in value and cuda in value:
            gain = calculate_gain(value, trt, cuda)
            value[trt_cuda_gain] = "{:.2f} %".format(gain)
        if trt_fp16 in value and cuda_fp16 in value:
            gain = calculate_gain(value, trt_fp16, cuda_fp16)
            value[trt_cuda_fp16_gain] = "{:.2f} %".format(gain)
        if trt in value and standalone_trt in value:
            gain = calculate_gain(value, trt, standalone_trt)
            value[trt_native_gain] = "{:.2f} %".format(gain)
        if trt_fp16 in value and standalone_trt_fp16 in value:
            gain = calculate_gain(value, trt_fp16, standalone_trt_fp16)
            value[trt_native_fp16_gain] = "{:.2f} %".format(gain)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ParseDictArgAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        dict_arg = {}

        for key_val in values.split(","):
            try:
                key, val = key_val.split("=")
            except ValueError:
                parser.error(f"argument {option_string}: Expected '=' between key and value")

            if key in dict_arg:
                parser.error(f"argument {option_string}: Specified duplicate key '{key}'")

            dict_arg[key] = val

        setattr(namespace, self.dest, dict_arg)


def parse_arguments():
    # Used by argparse to display usage information for custom inputs.
    dict_arg_metavar = "Opt1=Val1,Opt2=Val2..."

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--comparison",
        required=False,
        default="cuda_trt",
        choices=["cuda_trt", "acl"],
        help="EPs to compare: CPU vs. CUDA vs. TRT or CPU vs. ACL",
    )

    parser.add_argument(
        "-m",
        "--model_source",
        required=False,
        default="model_list.json",
        help="Model source: (1) model list file (2) model directory.",
    )

    parser.add_argument(
        "-r",
        "--running_mode",
        required=False,
        default="benchmark",
        choices=["validate", "benchmark"],
        help="Testing mode.",
    )

    parser.add_argument(
        "-i",
        "--input_data",
        required=False,
        default="fix",
        choices=["fix", "random"],
        help="Type of input data.",
    )

    parser.add_argument(
        "-o",
        "--perf_result_path",
        required=False,
        default="result",
        help="Directory for perf result.",
    )

    parser.add_argument(
        "-w",
        "--workspace",
        required=False,
        default="/",
        help="Workspace to find tensorrt and perf script (with models if parsing with model file)",
    )

    parser.add_argument(
        "-e",
        "--ep_list",
        nargs="+",
        required=False,
        default=None,
        help="Specify ORT Execution Providers list.",
    )

    parser.add_argument(
        "--trt_ep_options",
        required=False,
        default={
            "trt_engine_cache_enable": "True",
            "trt_max_workspace_size": "4294967296",
        },
        action=ParseDictArgAction,
        metavar=dict_arg_metavar,
        help="Specify options for the ORT TensorRT Execution Provider",
    )

    parser.add_argument(
        "--cuda_ep_options",
        required=False,
        default={},
        action=ParseDictArgAction,
        metavar=dict_arg_metavar,
        help="Specify options for the ORT CUDA Execution Provider",
    )

    parser.add_argument(
        "-z",
        "--track_memory",
        required=False,
        default=True,
        help="Track CUDA and TRT Memory Usage",
    )

    parser.add_argument("-b", "--io_binding", required=False, default=False, help="Bind Inputs")

    parser.add_argument(
        "-g",
        "--graph_enablement",
        required=False,
        default=enable_all,
        choices=[disable, basic, extended, enable_all],
        help="Choose graph optimization enablement.",
    )

    parser.add_argument("--ep", required=False, default=None, help="Specify ORT Execution Provider.")

    parser.add_argument(
        "--fp16",
        required=False,
        default=True,
        action="store_true",
        help="Inlcude Float16 into benchmarking.",
    )

    parser.add_argument("--trtexec", required=False, default=None, help="trtexec executable path.")

    # Validation options
    parser.add_argument(
        "--percent_mismatch",
        required=False,
        default=20.0,
        help="Allowed percentage of mismatched elements in validation.",
    )
    parser.add_argument(
        "--rtol",
        required=False,
        default=0,
        help="Relative tolerance for validating outputs.",
    )
    parser.add_argument(
        "--atol",
        required=False,
        default=20,
        help="Absolute tolerance for validating outputs.",
    )

    parser.add_argument(
        "-t",
        "--test_times",
        required=False,
        default=1,
        type=int,
        help="Number of repeat times to get average inference latency.",
    )

    parser.add_argument("--write_test_result", type=str2bool, required=False, default=True, help="")
    parser.add_argument("--benchmark_fail_csv", required=False, default=None, help="")
    parser.add_argument("--benchmark_success_csv", required=False, default=None, help="")
    parser.add_argument("--benchmark_latency_csv", required=False, default=None, help="")
    parser.add_argument("--benchmark_metrics_csv", required=False, default=None, help="")
    parser.add_argument("--system_info_csv", required=False, default=None, help="")

    args = parser.parse_args()

    return args


def setup_logger(verbose):
    if verbose:
        coloredlogs.install(
            level="DEBUG",
            fmt="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s",
        )
    else:
        coloredlogs.install(fmt="%(message)s")
        logging.getLogger("transformers").setLevel(logging.WARNING)


def parse_models_helper(args, models):
    model_source = os.path.join(args.workspace, args.model_source)
    if ".json" in model_source:
        logger.debug("Parsing model information from file ...")
        parse_models_info_from_file(args.workspace, model_source, models)
    else:
        logger.debug("Parsing model information from directory ...")
        parse_models_info_from_directory(model_source, models)


def main():
    args = parse_arguments()
    setup_logger(False)
    pp = pprint.PrettyPrinter(indent=4)

    logger.info("\n\nStart perf run ...\n")

    models = {}
    parse_models_helper(args, models)

    perf_start_time = datetime.now()
    (
        success_results,
        model_to_latency,
        model_to_fail_ep,
        model_to_op_metrics,
        model_to_session,
    ) = run_onnxruntime(args, models)
    perf_end_time = datetime.now()

    logger.info("Done running the perf.")
    logger.info("\nTotal time for benchmarking all models: %s", str(perf_end_time - perf_start_time))
    logger.info("\nTotal models: %d", len(models))

    fail_model_cnt = 0
    for key, value in models.items():
        if key in model_to_fail_ep:
            fail_model_cnt += 1
    logger.info("Fail models: %d", fail_model_cnt)
    logger.info("Success models: %d", len(models) - fail_model_cnt)

    path = os.path.join(os.getcwd(), args.perf_result_path)
    if not os.path.exists(path):
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if len(model_to_fail_ep) > 0:
        logger.info("\n============================================")
        logger.info("========== Failing Models/EPs ==============")
        logger.info("============================================")
        logger.info(model_to_fail_ep)
        write_map_to_file(model_to_fail_ep, FAIL_MODEL_FILE)

        if args.write_test_result:
            csv_filename = args.benchmark_fail_csv if args.benchmark_fail_csv else f"benchmark_fail_{time_stamp}.csv"
            csv_filename = os.path.join(path, csv_filename)
            output_fail(model_to_fail_ep, csv_filename)

    if len(model_to_latency) > 0:
        logger.info("\n==========================================")
        logger.info("=========== Models/EPs latency ===========")
        logger.info("==========================================")
        add_improvement_information(model_to_latency)
        pretty_print(pp, model_to_latency)
        write_map_to_file(model_to_latency, LATENCY_FILE)
        if args.write_test_result:
            csv_filename = (
                args.benchmark_latency_csv if args.benchmark_latency_csv else f"benchmark_latency_{time_stamp}.csv"
            )
            csv_filename = os.path.join(path, csv_filename)
            output_latency(model_to_latency, csv_filename)

    if success_results:
        csv_filename = (
            args.benchmark_success_csv if args.benchmark_success_csv else f"benchmark_success_{time_stamp}.csv"
        )
        csv_filename = os.path.join(path, csv_filename)
        output_details(success_results, csv_filename)

    if len(model_to_op_metrics) > 0:
        logger.info("\n=========================================")
        logger.info("========== Models/EPs metrics  ==========")
        logger.info("=========================================")
        pretty_print(pp, model_to_op_metrics)
        write_map_to_file(model_to_op_metrics, OP_METRICS_FILE)

        if args.write_test_result:
            csv_filename = (
                args.benchmark_metrics_csv if args.benchmark_metrics_csv else f"benchmark_metrics_{time_stamp}.csv"
            )
            csv_filename = os.path.join(path, csv_filename)
            output_metrics(model_to_op_metrics, csv_filename)

    if len(model_to_session) > 0:
        write_map_to_file(model_to_session, SESSION_FILE)


if __name__ == "__main__":
    main()
