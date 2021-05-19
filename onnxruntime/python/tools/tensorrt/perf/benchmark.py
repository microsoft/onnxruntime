import os
import csv
import timeit
from datetime import datetime
import numpy
import logging
import coloredlogs
import numpy as np
import argparse
import copy
import json
import re
import sys
import onnxruntime
from onnx import numpy_helper
from perf_utils import *
import pprint
import time
import pandas as pd
from float16 import *

debug = False
sys.path.append('.')
logger = logging.getLogger('')

# global ep variables 
cpu = "CPUExecutionProvider"
acl = "ACLExecutionProvider"
cuda = "CUDAExecutionProvider"
cuda_fp16 = "CUDAExecutionProvider_fp16"
trt = "TensorrtExecutionProvider"
trt_fp16 = "TensorrtExecutionProvider_fp16"
standalone_trt = "Standalone_TRT"
standalone_trt_fp16 = "Standalone_TRT_fp16"

ep_to_provider_list = {
    cpu: [cpu],
    acl: [acl], 
    cuda: [cuda],
    cuda_fp16: [cuda],
    trt: [trt, cuda],
    trt_fp16: [trt, cuda]
}

# latency gain headers 
trt_cuda_gain = 'TRT_CUDA_gain(%)'
trt_cuda_fp16_gain = 'TRT_CUDA_fp16_gain(%)'
trt_native_gain = 'TRT_Standalone_gain(%)'
trt_native_fp16_gain = 'TRT_Standalone_fp16_gain(%)'

# metadata
FAIL_MODEL_FILE = ".fail_model_map"
LATENCY_FILE = ".latency_map"
METRICS_FILE = ".metrics_map"
MEMORY_FILE = './temp_memory.csv'

def run_trt_standalone(trtexec, model_path, ort_inputs, all_inputs_shape, fp16):
    logger.info("running standalone trt")
    model_path = "--onnx=" + model_path
    input_shape = []

    logger.info(all_inputs_shape)

    for i in range(len(ort_inputs)):
        name = ort_inputs[i].name

        shape = []
        for j in all_inputs_shape[i]:
            shape.append(str(j))
        shape = "x".join(shape)
        shape = name + ':' + shape
        input_shape.append(shape)

    shapes_arg = '--optShapes=' + ','.join(input_shape)
    logger.info(shapes_arg)

    result = {}

    if fp16:
        out = get_output([trtexec, model_path, "--fp16", "--percentile=90", "--explicitBatch", shapes_arg])
    else:
        out = get_output([trtexec, model_path, "--percentile=90", "--explicitBatch", shapes_arg])

    tmp = out.split("\n")
    target_list = []
    for t in tmp:
        if 'mean:' in t:
            target_list.append(t)

        if 'percentile:' in t:
            target_list.append(t)

    target = target_list[2]
    start = target.find('mean:') + 6
    end = target.find('ms')
    result["average_latency_ms"] = target[start:end]

    target = target_list[3]
    start = target.find('percentile:') + 12
    end = target.find('ms')
    result["latency_90_percentile"] = target[start:end]

    logger.info(result)
    return result

def get_latency_result(runtimes, batch_size, mem_mb=None):
    latency_ms = sum(runtimes) / float(len(runtimes)) * 1000.0
    latency_variance = numpy.var(runtimes, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    result = {
        "test_times": len(runtimes),
        "latency_variance": "{:.2f}".format(latency_variance),
        "latency_90_percentile": "{:.2f}".format(numpy.percentile(runtimes, 90) * 1000.0),
        "latency_95_percentile": "{:.2f}".format(numpy.percentile(runtimes, 95) * 1000.0),
        "latency_99_percentile": "{:.2f}".format(numpy.percentile(runtimes, 99) * 1000.0),
        "average_latency_ms": "{:.2f}".format(latency_ms),
        "QPS": "{:.2f}".format(throughput),
    }
    if mem_mb:
        result.update({"memory":mem_mb})
    return result


def get_ort_session_inputs_and_outputs(name, session, ort_input):

    sess_inputs = {}
    sess_outputs = None

    if 'bert_squad' in name.lower() or 'bert-squad' in name.lower():
        unique_ids_raw_output = ort_input[0]
        input_ids = ort_input[1]
        input_mask = ort_input[2]
        segment_ids = ort_input[3]

        sess_inputs = {
                "unique_ids_raw_output___9:0": unique_ids_raw_output,
                "input_ids:0": input_ids[0:1],
                "input_mask:0": input_mask[0:1],
                "segment_ids:0": segment_ids[0:1]}
        sess_outputs = ["unique_ids:0", "unstack:0", "unstack:1"]

    elif 'bidaf' in name.lower():
        sess_inputs = {
                "context_word": ort_input[0],
                "context_char": ort_input[2],
                "query_word": ort_input[1],
                "query_char": ort_input[3]}
        sess_outputs = ["start_pos","end_pos"]

    elif 'yolov4' in name.lower():
        sess_inputs[session.get_inputs()[0].name] = ort_input[0]
        sess_outputs = ['Identity:0']

    elif 'shufflenet-v2' in name.lower() or 'shufflenet_v2' in name.lower():
        sess_inputs[session.get_inputs()[0].name] = ort_input

    else:
        sess_inputs = {}
        for i in range(len(session.get_inputs())):
            sess_inputs[session.get_inputs()[i].name] = ort_input[i]

    return (sess_inputs, sess_outputs)

def track_ep_memory(ep): 
     return trt in ep or cuda in ep or standalone_trt in ep

def get_trtexec_pid(df, python_pid): 
    for pid in df['pid'].tolist(): 
        if pid != python_pid: 
            return pid

def get_max_memory(trtexec): 
    df = pd.read_csv(MEMORY_FILE)
    pid = df['pid'].iloc[0]
    
    if trtexec: 
        pid = get_trtexec_pid(df, pid) 
    
    mem_series = df.loc[df['pid'] == pid, ' used_gpu_memory [MiB]']
    max_mem = max(mem_series.str.replace(' MiB','').astype(int))
    return max_mem

def start_memory_tracking(): 
    p = subprocess.Popen(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv", "-l", "1", "-f", MEMORY_FILE])
    return p

def end_memory_tracking(p, trtexec): 
    p.terminate()
    p.wait()
    mem_usage = get_max_memory(trtexec) 
    os.remove(MEMORY_FILE)
    return mem_usage

def inference_ort(args, name, session, ep, ort_inputs, result_template, repeat_times, batch_size):
    runtimes = []
    if args.input_data == "random":
        repeat_times = 1 # warn-up run is included in ort_inputs
    else:
        repeat_times += 1 # add warn-up run
    
    mem_usage = None
    for ort_input in ort_inputs:
        sess_inputs, sess_outputs = get_ort_session_inputs_and_outputs(name, session, ort_input)
        if debug:
            logger.info("ORT session inputs:")
            logger.info(sess_inputs)
            logger.info("ORT session outputs:")
            logger.info(sess_outputs)

        try:
            if args.track_memory and track_ep_memory(ep): 

                p = start_memory_tracking()            
                runtime = timeit.repeat(lambda: session.run(sess_outputs, sess_inputs), number=1, repeat=repeat_times)
                mem_usage = end_memory_tracking(p, False)
            else: 
                runtime = timeit.repeat(lambda: session.run(sess_outputs, sess_inputs), number=1, repeat=repeat_times)

            runtimes += runtime[1:] # remove warmup

        except Exception as e:
            logger.error(e)
            return None

    logger.info(runtimes)

    result = {}
    result.update(result_template)
    result.update({"io_binding": False})
    latency_result = get_latency_result(runtimes, batch_size, mem_usage)
    result.update(latency_result)
    logger.info(result)
    return result

def inference_ort_and_get_prediction(name, session, ort_inputs):

    ort_outputs = []
    for ort_input in ort_inputs:
        sess_inputs, sess_outputs = get_ort_session_inputs_and_outputs(name, session, ort_input)
        if debug:
            logger.info("ORT session inputs:")
            logger.info(sess_inputs)
            logger.info("ORT session outputs:")
            logger.info(sess_outputs)

        result = session.run(sess_outputs, sess_inputs)
        
        if debug:
            logger.info("ORT session output results:")
            logger.info(result)

        # handle shape of output differently
        if 'bert_squad' in name.lower():
            ort_outputs.append([result])
        elif 'shufflenet-v2' in name.lower() or 'shufflenet_v2' in name.lower():
            ort_outputs.append(result[0])
        else:
            ort_outputs.append(result)

    return ort_outputs

def get_acl_version():
    from pathlib import Path
    home = str(Path.home())
    p = subprocess.run(["find", home, "-name", "libarm_compute.so"], check=True, stdout=subprocess.PIPE)
    libarm_compute_path = p.stdout.decode("ascii").strip()
    if libarm_compute_path == '':
        return "No Compute Library Found"
    else:
        p = subprocess.run(["strings", libarm_compute_path], check=True, stdout=subprocess.PIPE)  
        libarm_so_strings = p.stdout.decode("ascii").strip()
        version_match = re.search(r'arm_compute_version.*\n', libarm_so_strings)
        version = version_match.group(0).split(' ')[0]
        return version

#######################################################################################################################################
# The following two lists will be generated.
#
# inputs: [[test_data_0_input_0.pb, test_data_0_input_1.pb ...], [test_data_1_input_0.pb, test_data_1_input_1.pb ...] ...]
# outputs: [[test_data_0_output_0.pb, test_data_0_output_1.pb ...], [test_data_1_output_0.pb, test_data_1_output_1.pb ...] ...]
#######################################################################################################################################
def load_onnx_model_zoo_test_data(path, all_inputs_shape, data_type="fp32"):
    logger.info("Parsing test data in {} ...".format(path))
    p1 = subprocess.Popen(["find", path, "-name", "test_data*", "-type", "d"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    test_data_set_dir = stdout.split("\n")
    logger.info(test_data_set_dir)

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

        # load inputs
        p1 = subprocess.Popen(["find", ".", "-name", "input*"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()
        input_data = stdout.split("\n")
        logger.info(input_data)

        input_data_pb = []
        for data in input_data:
            tensor = onnx.TensorProto()
            with open(data, 'rb') as f:
                tensor.ParseFromString(f.read())

                tensor_to_array = numpy_helper.to_array(tensor)

                if data_type == "fp16" and tensor_to_array.dtype == np.dtype(np.float32):
                    tensor_to_array = tensor_to_array.astype(np.float16)
                input_data_pb.append(tensor_to_array)
                print(input_data_pb[0].shape)
                if not shape_flag:
                    all_inputs_shape.append(input_data_pb[-1].shape)
                logger.info(all_inputs_shape[-1])
        inputs.append(input_data_pb)
        print(input_data_pb[0].shape)
        print("HERE")
        logger.info('Loaded {} inputs successfully.'.format(len(inputs)))

        # load outputs
        p1 = subprocess.Popen(["find", ".", "-name", "output*"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()
        output_data = stdout.split("\n")
        logger.info(output_data)

        if len(output_data) > 0 and output_data[0] != '':
            output_data_pb = []
            for data in output_data:
                tensor = onnx.TensorProto()
                with open(data, 'rb') as f:
                    tensor.ParseFromString(f.read())

                    tensor_to_array = numpy_helper.to_array(tensor)

                    if data_type == "fp16" and tensor_to_array.dtype == np.dtype(np.float32):
                        tensor_to_array = tensor_to_array.astype(np.float16)
                    output_data_pb.append(tensor_to_array)

                    logger.info(np.array(output_data_pb[-1]).shape)
            outputs.append(output_data_pb)
            logger.info('Loaded {} outputs successfully.'.format(len(outputs)))

        os.chdir(pwd)

    return inputs, outputs

def generate_onnx_model_random_input(test_times, ref_input):
    inputs = []

    for i in range(test_times):

        input_data = []
        for tensor in ref_input:
            shape = tensor.shape
            dtype = tensor.dtype
            if dtype == np.int8 or   \
               dtype == np.uint8 or  \
               dtype == np.int16 or  \
               dtype == np.uint16 or \
               dtype == np.int32 or  \
               dtype == np.uint32 or \
               dtype == np.int64 or  \
               dtype == np.uint64:
                new_tensor = np.random.randint(0, np.max(tensor)+1, shape, dtype)
            else:
                new_tensor = np.random.random_sample(shape).astype(dtype)

            if debug:
                logger.info("original tensor:")
                logger.info(tensor)
                logger.info("new random tensor:")
                logger.info(new_tensor)
                logger.info("\n")

            input_data.append(new_tensor)
        inputs.append(input_data)

    return inputs

def percentage_in_allowed_threshold(e, percent_mismatch):
    percent_string = re.search(r'\(([^)]+)', str(e)).group(1)
    if "%" in percent_string:
        percentage_wrong = float(percent_string.replace("%",""))
        return percentage_wrong < percent_mismatch
    else: 
        return False # error in output 

def validate(all_ref_outputs, all_outputs, rtol, atol, percent_mismatch):
    if len(all_ref_outputs) == 0:
        logger.info("No reference output provided.")
        return True, None

    logger.info('Reference {} results.'.format(len(all_ref_outputs)))
    logger.info('Predicted {} results.'.format(len(all_outputs)))
    logger.info('rtol: {}, atol: {}'.format(rtol, atol))

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
                except Exception as e:
                    if percentage_in_allowed_threshold(e, percent_mismatch):    
                        continue
                    logger.error(e)
                    return False, e

    logger.info('ONNX Runtime outputs are similar to reference outputs!')
    return True, None

# not use for this script
def cleanup_files():
    files = []
    p = subprocess.Popen(["find", ".", "-name", "test_data_set*", "-type", "d"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    files = files + stdout.split("\n")

    p = subprocess.Popen(["find", ".", "-name", "*.onnx"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    files = files + stdout.split("\n")

    p = subprocess.Popen(["find", ".", "-name", "*.gz"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    files = files + stdout.split("\n")

    for f in files:
        if "custom_test_data" in f:
            logger.info(f)
            continue
        subprocess.Popen(["rm","-rf", f], stdout=subprocess.PIPE)

def remove_profiling_files(path):
    files = []
    out = get_output(["find", path, "-name", "onnxruntime_profile*"])
    files = files + out.split("\n")

    for f in files:
        if "custom_test_data" in f:
            continue
        subprocess.Popen(["sudo","rm","-rf", f], stdout=subprocess.PIPE)


def update_fail_report(fail_results, model, ep, e_type, e):
    result = {}

    result["model"] = model
    result["ep"] = ep
    result["error type"] = e_type
    result["error message"] = re.sub('^\n', '', str(e))

    fail_results.append(result)

def update_metrics_map(model_to_metrics, model_name, ep_to_operator):
    if len(ep_to_operator) <= 0:
        return

    if model_name not in model_to_metrics:
        model_to_metrics[model_name] = {}

    for ep, op_map in ep_to_operator.items():
        if ep not in model_to_metrics[model_name]:
            model_to_metrics[model_name][ep] = {}

        if ep == cuda or ep == cuda_fp16:
            model_to_metrics[model_name][ep]['ratio_of_ops_in_cuda_not_fallback_cpu'] = calculate_cuda_op_percentage(op_map) 
            model_to_metrics[model_name][ep]['total_ops'] = get_total_ops(op_map) 
        else:
            total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(op_map)
            model_to_metrics[model_name][ep]['total_ops'] = get_total_ops(op_map) 
            model_to_metrics[model_name][ep]['total_trt_execution_time'] = total_trt_execution_time
            model_to_metrics[model_name][ep]['total_execution_time'] = total_execution_time
            model_to_metrics[model_name][ep]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt


def update_metrics_map_ori(model_to_metrics, name, ep_to_operator):
    if len(ep_to_operator) <= 0:
        return

    trt_op_map = None
    trt_fp16_op_map = None
    cuda_op_map = None
    cuda_fp16_op_map = None

    for ep, op_map in ep_to_operator.items():
        if ep == cuda:
            cuda_op_map = op_map
        elif ep == cuda_fp16:
            cuda_fp16_op_map = op_map
        elif ep == trt:
            trt_op_map = op_map
        elif ep == trt_fp16:
            trt_fp16_op_map = op_map


    if name not in model_to_metrics:
        model_to_metrics[name] = {}

    if cuda_op_map:
        model_to_metrics[name]['ratio_of_ops_in_cuda_not_fallback_cpu'] = calculate_cuda_op_percentage(cuda_op_map) 

    if trt_op_map:
        total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_op_map)
        model_to_metrics[name]['total_trt_execution_time'] = total_trt_execution_time
        model_to_metrics[name]['total_execution_time'] = total_execution_time
        model_to_metrics[name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt
        if cuda_op_map:
            total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_op_map, cuda_op_map)
            model_to_metrics[name]['total_ops_in_trt'] = total_ops_in_trt
            model_to_metrics[name]['total_ops'] = total_ops
            model_to_metrics[name]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt

    if trt_fp16_op_map:
        total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_fp16_op_map)
        name_ = name + " (FP16)"
        model_to_metrics[name_] = {}
        model_to_metrics[name_]['total_trt_execution_time'] = total_trt_execution_time
        model_to_metrics[name_]['total_execution_time'] = total_execution_time
        model_to_metrics[name_]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt
        if cuda_fp16_op_map:
            total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_fp16_op_map, cuda_op_map)
            model_to_metrics[name_]['total_ops_in_trt'] = total_ops_in_trt
            model_to_metrics[name_]['total_ops'] = total_ops
            model_to_metrics[name_]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt

    if debug:
        pp = pprint.PrettyPrinter(indent=4)
        logger.info('CUDA operator map:')
        pp.pprint(cuda_op_map)
        logger.info('TRT operator map:')
        pp.pprint(trt_op_map)
        logger.info('CUDA FP16 operator map:')
        pp.pprint(cuda_fp16_op_map)
        logger.info('TRT FP16 operator map:')
        pp.pprint(trt_fp16_op_map)


###################################################################################################
#
# model: {ep1: {error_type: xxx, error_message: xxx}, ep2: {error_type: xx, error_message: xx}}
#
###################################################################################################
def update_fail_model_map(model_to_fail_ep, model_name, ep, e_type, e):

    if model_name in model_to_fail_ep and ep in model_to_fail_ep[model_name]:
        return

    if model_name not in model_to_fail_ep:
        model_to_fail_ep[model_name] = {} 

    new_map = {}
    new_map["error_type"] = e_type
    new_map["error_message"] = re.sub('^\n', '', str(e))
    model_to_fail_ep[model_name][ep] = new_map

    # If TRT fails, TRT FP16 should fail as well
    if ep == trt:
        ep_ = trt_fp16
        e_ = "skip benchmarking since TRT failed already."
        new_map_1 = {}
        new_map_1["error_type"] = e_type
        new_map_1["error_message"] = e_
        model_to_fail_ep[model_name][ep_] = new_map_1 

def update_fail_model_map_ori(model_to_fail_ep, fail_results, model_name, ep, e_type, e):

    if model_name in model_to_fail_ep and ep in model_to_fail_ep[model_name]:
        return

    if model_name not in model_to_fail_ep:
        model_to_fail_ep[model_name] = {} 
    
    model_to_fail_ep[model_name][ep] = e_type
    update_fail_report(fail_results, model_name, ep, e_type, e)

    # If TRT fails, TRT FP16 should fail as well
    if ep == trt:
        ep_ = trt_fp16
        error_message_ = "skip benchmarking since TRT failed already."
        update_fail_report(fail_results, model_name, ep_, e_type, error_message_)
        model_to_fail_ep[model_name][ep_] = e_type

def skip_ep(model_name, ep, model_to_fail_ep):

    if model_name not in model_to_fail_ep:
        return False

    fail_ep_list = model_to_fail_ep[model_name]

    # if ep in fail_ep_list and fail_ep_list[ep] == "runtime error":
    if ep in fail_ep_list:
        logger.info("Skip testing " + model_name + " using " + ep + " since it has some issues.")
        return True

    return False

def read_map_from_file(map_file):
    with open(map_file) as f:
        try:
            data = json.load(f)
        except Exception as e:
            return None

    return data

def write_map_to_file(result, file_name):
    existed_result = {}
    if os.path.exists(file_name):
        existed_result = read_map_from_file(file_name)
    
    for model, ep_list in result.items():
        if model in existed_result:
            existed_result[model] = {** existed_result[model], ** result[model]} 
        else:
            existed_result[model] = result[model]

    with open(file_name, 'w') as file:
        file.write(json.dumps(existed_result)) # use `json.loads` to do the reverse


def get_cuda_version():
    nvidia_strings = get_output(["nvidia-smi"]) 
    version = re.search(r'CUDA Version: \d\d\.\d', nvidia_strings).group(0) 
    return version
    
def get_trt_version(workspace):
    libnvinfer = get_output(["find", workspace, "-name", "libnvinfer.so.*"])
    nvinfer = re.search(r'.*libnvinfer.so.*', libnvinfer).group(0)
    trt_strings = get_output(["nm", "-D", nvinfer])
    version = re.search(r'tensorrt_version.*', trt_strings).group(0)
    return version
 
def get_linux_distro(): 
    linux_strings = get_output(["cat", "/etc/os-release"])
    stdout = linux_strings.split("\n")[:2]
    infos = []
    for row in stdout:
        row = re.sub('=', ':  ', row)
        row = re.sub('"', '', row)
        infos.append(row)
    return infos 

def get_memory_info():
    mem_strings = get_output(["cat", "/proc/meminfo"])
    stdout = mem_strings.split("\n")
    infos = []
    for row in stdout:
        if "Mem" in row:
            row = re.sub(': +', ':  ', row)
            infos.append(row)
    return infos

def get_cpu_info(): 
    cpu_strings = get_output(["lscpu"])
    stdout = cpu_strings.split("\n")
    infos = []
    for row in stdout:
        if "mode" in row or "Arch" in row or "name" in row:
            row = re.sub(': +', ':  ', row)
            infos.append(row)
    return infos

def get_gpu_info():
    info = get_output(["lspci", "-v"])
    infos = re.findall('NVIDIA.*', info)
    return infos

def get_system_info(workspace):
    info = {}
    info["cuda"] = get_cuda_version()
    info["trt"] = get_trt_version(workspace)
    info["linux_distro"] = get_linux_distro()
    info["cpu_info"] = get_cpu_info()
    info["gpu_info"] = get_gpu_info()
    info["memory"] = get_memory_info()

    return info

def find_model_path(path):
    p1 = subprocess.Popen(["find", path, "-name", "*.onnx"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("ascii").strip()
    model_path = stdout.split("\n")
    logger.info(model_path)

    if model_path == ['']:
        return None

    target_model_path = []
    for m in model_path:
        if "by_trt_perf" in m or m.startswith('.'):
            continue
        target_model_path.append(m)

    logger.info(target_model_path)
    if len(target_model_path) > 1:
        logger.error("We expect to find only one model in " + path)
        raise

    return target_model_path[0]

def find_model_directory(path):
    p1 = subprocess.Popen(["find", path, "-maxdepth", "1", "-mindepth", "1", "-name", "*", "-type", "d"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("ascii").strip()
    model_dir = stdout.split("\n")
    # print(model_dir)

    if model_dir == ['']:
        return None

    return model_dir

def find_test_data_directory(path):
    p1 = subprocess.Popen(["find", path, "-maxdepth", "1", "-name", "test_data*", "-type", "d"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    test_data_dir = stdout.split("\n")
    logger.info(test_data_dir)

    if test_data_dir == ['']:
        return None

    return test_data_dir

def parse_models_info_from_directory(path, models):

    test_data_dir = find_test_data_directory(path) 

    if test_data_dir:
        model_name = os.path.split(path)[-1]
        model_name = model_name + '_' + os.path.split(os.path.split(path)[0])[-1] # get opset version as model_name
        model_path = find_model_path(path)

        model = {}
        model["model_name"] = model_name
        model["model_path"] = model_path 
        model["working_directory"] = path 
        model["test_data_path"] = path 

        models[model_name] = model 

        logger.info(model)
        return
    
    model_dir = find_model_directory(path)
    
    if model_dir:
        for dir in model_dir:
            parse_models_info_from_directory(os.path.join(path, dir), models)
    

def parse_models_info_from_file(root_dir, path, models):

    # default working directory
    root_working_directory = root_dir

    with open(path) as f:
        data = json.load(f)

        for row in data:

            if 'root_working_directory' in row:
                root_working_directory = row['root_working_directory']
                continue

            if 'model_name' in row:
                models[row['model_name']] = {}
            else:
                logger.error('Model name must be provided in models_info.json')
                raise

            model = models[row['model_name']]

            if 'working_directory' in row:
                if os.path.isabs(row['working_directory']):
                    model['working_directory'] = row['working_directory']
                else:
                    model['working_directory'] = os.path.join(root_working_directory, row['working_directory'])
            else:
                logger.error('Model path must be provided in models_info.json')
                raise

            if 'model_path' in row:
                model['model_path'] = row['model_path']
            else:
                logger.error('Model path must be provided in models_info.json')
                raise

            if 'test_data_path' in row:
                model['test_data_path'] = row['test_data_path']
            else:
                logger.error('Test data path must be provided in models_info.json')
                raise

            if 'model_path_fp16' in row:
                model['model_path_fp16'] = row['model_path_fp16']

            if 'test_data_path_fp16' in row:
                model['test_data_path_fp16'] = row['test_data_path_fp16']


def convert_model_from_float_to_float16(model_path):
    # from onnxmltools.utils.float16_converter import convert_float_to_float16
    from onnxmltools.utils import load_model, save_model
    from float16 import convert_float_to_float16

    onnx_model = load_model(model_path)
    new_onnx_model = convert_float_to_float16(onnx_model)
    save_model(new_onnx_model, 'new_fp16_model_by_trt_perf.onnx')

    return os.path.join(os.getcwd(), "new_fp16_model_by_trt_perf.onnx")

def get_test_data(fp16, test_data_dir, all_inputs_shape):
    inputs = []
    ref_outputs = []

    # read input/output of test data
    if fp16:
        inputs, ref_outputs = load_onnx_model_zoo_test_data(test_data_dir, all_inputs_shape, "fp16")
    else:
        inputs, ref_outputs = load_onnx_model_zoo_test_data(test_data_dir, all_inputs_shape)

    return inputs, ref_outputs

def create_session(model_path, providers, session_options):

    logger.info(model_path)
    try:
        print("creating session") 
        session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=session_options)

        return session
    except Exception as e:
        if "shape inference" in e:
            logger.info("Use symbolic_shape_infer.py")

            new_model_path = model_path[:].replace(".onnx", "_new_by_trt_perf.onnx")
            exec = os.environ["SYMBOLIC_SHAPE_INFER"]
            logger.info(exec)

            if not os.path.exists(new_model_path):
                subprocess.run("python3 " + exec +" --input " + model_path + " --output " + new_model_path + " --auto_merge", shell=True, check=True)
            print("symbolic creating session") 
            session = onnxruntime.InferenceSession(new_model_path, providers=providers, sess_options=session_options)

            return session
        else: 
            raise Exception(e) 

def run_onnxruntime(args, models):

    success_results = []
    model_to_latency = {} # model -> cuda and tensorrt latency
    model_to_metrics = {} # model -> metrics from profiling file
    model_to_fail_ep = {} # model -> failing ep

    ep_list = []
    if args.ep:
        ep_list.append(args.ep)
    else:
        if args.fp16:
            ep_list = [cpu, cuda, trt, cuda_fp16, trt_fp16]
        else:
            ep_list = [cpu, cuda, trt]

    validation_exemption = [trt_fp16]


    if os.path.exists(FAIL_MODEL_FILE):
        model_to_fail_ep = read_map_from_file(FAIL_MODEL_FILE)

    #######################
    # iterate model
    #######################
    for name, model_info in models.items():
        latency_result = {}
        path = model_info["working_directory"]

        pwd = os.getcwd()
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        path = os.getcwd()

        if args.running_mode == "validate": 
            remove_profiling_files(path)
        
        inputs = []
        ref_outputs = []
        all_inputs_shape = [] # use for standalone trt
        ep_to_operator = {} # ep -> { operator -> count }
        profile_already_parsed = set()


        #######################
        # iterate ep
        #######################
        for ep in ep_list:

            if skip_ep(name, ep, model_to_fail_ep):
                continue

            ep_ = ep_to_provider_list[ep][0]
            if (ep_ not in onnxruntime.get_available_providers()):
                logger.error("No {} support".format(ep_))
                continue

            model_path = model_info["model_path"]
            test_data_dir = model_info["test_data_path"]

            if ep == cuda_fp16:
                logger.info("[Initialize]  model = {}, ep = {} ,FP16 = True ...".format(name, ep))
                fp16 = True
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

                # handle model
                if "model_path_fp16" in model_info:
                    model_path = model_info["model_path_fp16"]

                else:
                    try:
                        model_path = convert_model_from_float_to_float16(model_path)

                    except Exception as e:
                        logger.error(e)
                        update_fail_model_map(model_to_fail_ep, name, ep, 'script error', e)
                        continue

                # handle test data
                if "test_data_path_fp16" in model_info:
                    test_data_dir = model_info["test_data_path_fp16"]
                    inputs, ref_outputs = get_test_data(False, test_data_dir, all_inputs_shape)
                else:
                    inputs, ref_outputs = get_test_data(True, test_data_dir, all_inputs_shape)
            
            elif ep == trt_fp16:
                logger.info("[Initialize]  model = {}, ep = {} ,FP16 = True ...".format(name, ep))
                fp16 = True
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

                inputs, ref_outputs = get_test_data(False, test_data_dir, all_inputs_shape)
            else:
                logger.info("[Initialize]  model = {}, ep = {} ,FP16 = False ...".format(name, ep))
                fp16 = False
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"

                inputs, ref_outputs = get_test_data(False, test_data_dir, all_inputs_shape)


            # generate random input data
            if args.input_data == "random":
                inputs = generate_onnx_model_random_input(args.test_times+1, inputs[0])

            #######################################
            # benchmark or validation
            #######################################
            if args.running_mode == 'benchmark':
                logger.info("\n----------------------------- benchmark -------------------------------------")

                options = onnxruntime.SessionOptions()
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

                # create onnxruntime inference session
                try:
                    sess = create_session(model_path, ep_to_provider_list[ep], options)

                except Exception as e:
                    logger.error(e)
                    continue

                logger.info("start to inference {} with {} ...".format(name, ep))
                logger.info(sess.get_providers())

                if sess:
                    logger.info("Model inputs nodes:")
                    for input_meta in sess.get_inputs():
                        logger.info(input_meta)
                    logger.info("Model outputs nodes:")
                    for output_meta in sess.get_outputs():
                        logger.info(output_meta)

                batch_size = 1
                result_template = {
                    "engine": "onnxruntime",
                    "version": onnxruntime.__version__,
                    "device": ep,
                    "fp16": fp16,
                    "io_binding": False,
                    "model_name": name,
                    "inputs": len(sess.get_inputs()),
                    "batch_size": batch_size,
                    "sequence_length": 1,
                    "datetime": str(datetime.now()),}
                    
                if trt in ep and args.trtexec:
                    
                    # get standalone TensorRT perf
                    try: 
                        ep = standalone_trt_fp16 if fp16 else standalone_trt
                        
                        if args.track_memory: 
                            p = start_memory_tracking()            
                            result = run_trt_standalone(args.trtexec, model_path, sess.get_inputs(), all_inputs_shape, fp16)
                            mem_usage = end_memory_tracking(p, True)
                            if result and mem_usage: 
                                result["memory"] = mem_usage

                        else: 
                            result = run_trt_standalone(args.trtexec, model_path, sess.get_inputs(), all_inputs_shape, fp16)
                    except Exception as e: 
                        logger.error(e)
                        update_fail_model_map(model_to_fail_ep, name, ep, 'runtime error', e)
                        continue

                else: 
                    result = inference_ort(args, name, sess, ep, inputs, result_template, args.test_times, batch_size)
                
                if result:

                    latency_result[ep] = {}
                    latency_result[ep]["average_latency_ms"] = result["average_latency_ms"]
                    latency_result[ep]["latency_90_percentile"] = result["latency_90_percentile"]
                    if "memory" in result: 
                        mem_usage = result.pop("memory")
                        latency_result[ep]["memory"] = mem_usage

                    if not args.trtexec: # skip standalone
                        success_results.append(result)

                    model_to_latency[name] = copy.deepcopy(latency_result)

                logger.info("---------------------------- benchmark [end] ----------------------------------\n")



            elif args.running_mode == 'validate':
                logger.info("\n----------------------------- validate -------------------------------------")

                # enable profiling to generate profiling file for analysis
                options = onnxruntime.SessionOptions()
                options.enable_profiling = True
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                time.sleep(1) # avoid to generate same profile file name

                # create onnxruntime inference session
                try:
                    sess = create_session(model_path, ep_to_provider_list[ep], options)

                except Exception as e:
                    logger.error(e)
                    update_fail_model_map(model_to_fail_ep, name, ep, 'runtime error', e)
                    continue

                sess.disable_fallback()

                logger.info("start to inference {} with {} ...".format(name, ep))
                logger.info(sess.get_providers())

                if sess:
                    logger.info("Model inputs nodes:")
                    for input_meta in sess.get_inputs():
                        logger.info(input_meta)
                    logger.info("Model outputs nodes:")
                    for output_meta in sess.get_outputs():
                        logger.info(output_meta)

                # run inference and validate the result
                #
                # currently skip TensorRT float16 validation intentionally
                if ep not in validation_exemption:
                    try:
                        ort_outputs = inference_ort_and_get_prediction(name, sess, inputs)

                        status = validate(ref_outputs, ort_outputs, args.rtol, args.atol, args.percent_mismatch)
                        if not status[0]:
                            update_fail_model_map(model_to_fail_ep, name, ep, 'result accuracy issue', status[1])
                            continue
                    except Exception as e:
                        logger.error(e)
                        update_fail_model_map(model_to_fail_ep, name, ep, 'runtime error', e)
                        continue

                    # Run inference again. the reason is that some ep like tensorrt
                    # it takes much longer time to generate graph on first run and
                    # we need to skip the perf result of that expensive run.
                    inference_ort_and_get_prediction(name, sess, inputs)
                else:
                    inference_ort_and_get_prediction(name, sess, inputs)
                    inference_ort_and_get_prediction(name, sess, inputs)

                sess.end_profiling()

                # get metrics from profiling file
                metrics = get_profile_metrics(path, profile_already_parsed, logger)
                if metrics:
                    logger.info(ep)
                    ep_to_operator[ep] = metrics

                logger.info("---------------------------- validate [end] ----------------------------------\n")

        ####################
        # end of iterate ep
        ####################


        # get percentage of execution time and operators in TRT
        update_metrics_map(model_to_metrics, name, ep_to_operator)

        # cleanup_files()
        os.chdir(pwd)

        # end of model

    return success_results, model_to_latency, model_to_fail_ep, model_to_metrics

def calculate_gain(value, ep1, ep2): 
    ep1_latency = float(value[ep1]['average_latency_ms'])
    ep2_latency = float(value[ep2]['average_latency_ms'])
    gain = (ep2_latency - ep1_latency)*100/ep2_latency
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

def output_details(results, csv_filename):
    need_write_header = True 
    if os.path.exists(csv_filename):
        need_write_header = False 

    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "fp16", "io_binding", "model_name", "inputs", "batch_size",
            "sequence_length", "datetime", "test_times", "QPS", "average_latency_ms", "latency_variance",
            "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        if need_write_header:
            csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Detail results are saved to csv file: {csv_filename}")

def output_fail(model_to_fail_ep, csv_filename):

    with open(csv_filename, mode="w", newline='') as csv_file:
        column_names = ["model", "ep", "error type", "error message"]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for model, model_info in model_to_fail_ep.items():
            for ep, ep_info in model_info.items():
                result = {}
                result["model"] = model
                result["ep"] = ep
                result["error type"] = ep_info["error_type"]
                result["error message"] = ep_info["error_message"]
                csv_writer.writerow(result)

    logger.info(f"Failing results are saved to csv file: {csv_filename}")
    
def read_success_from_file(success_file):
    success_results = []
    with open(success_file) as success:
       csv_reader = csv.DictReader(success)
       for row in csv_reader: 
           success_results.append(row)

    success_json = json.loads(json.dumps(success_results, indent=4))
    return success_json

def add_status_dict(status_dict, model_name, ep, status): 
    if model_name not in status_dict:
        status_dict[model_name] = {}
    status_dict[model_name][ep] = status

def build_status(status_dict, results, is_fail):
        
        if is_fail:
            for model, model_info in results.items():
                for ep, ep_info in model_info.items(): 
                    model_name = model
                    ep = ep
                    status = 'Fail'
                    add_status_dict(status_dict, model_name, ep, status)
        else: 
            for model, value in results.items():
                for ep, ep_info in value.items(): 
                    model_name = model
                    ep = ep
                    status = 'Pass'
                    add_status_dict(status_dict, model_name, ep, status)

        return status_dict

def output_status(results, csv_filename):
    
    need_write_header = True 
    if os.path.exists(csv_filename):
        need_write_header = False 

    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["Model",
                        cpu,
                        cuda + " fp32",
                        trt + " fp32",
                        standalone_trt + " fp32",
                        cuda + " fp16",
                        trt + " fp16",
                        standalone_trt + " fp16"
                        ]

        csv_writer = csv.writer(csv_file)

        if need_write_header:
            csv_writer.writerow(column_names)
    
        cpu_status = ""
        cuda_fp32_status = ""
        trt_fp32_status = ""
        standalone_fp32_status = ""
        cuda_fp16_status = ""
        trt_fp16_status = ""
        standalone_fp16_status = ""
        

        for model_name, ep_dict in results.items():
            for ep, status in ep_dict.items():
                if ep == cpu: 
                    cpu_status = status 
                elif ep == cuda: 
                    cuda_fp32_status = status
                elif ep == trt: 
                    trt_fp32_status = status
                elif ep == standalone_trt:
                    standalone_fp32_status = status
                elif ep == cuda_fp16: 
                    cuda_fp16_status = status
                elif ep == trt_fp16:
                    trt_fp16_status = status
                elif ep == standalone_trt_fp16: 
                    standalone_fp16_status = status
                else: 
                    continue
                    
            row = [model_name,
                   cpu_status, 
                   cuda_fp32_status, 
                   trt_fp32_status, 
                   standalone_fp32_status, 
                   cuda_fp16_status, 
                   trt_fp16_status, 
                   standalone_fp16_status]
            csv_writer.writerow(row)

def output_latency(results, csv_filename):
    need_write_header = True 
    if os.path.exists(csv_filename):
        need_write_header = False 

    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["Model",
                        "CPU fp32 \nmean (ms)",
                        "CPU fp32 \n 90th percentile (ms)",
                        "CUDA fp32 \nmean (ms)",
                        "CUDA fp32 \n90th percentile (ms)",
                        "CUDA EP fp32 \npeak memory usage (MiB)",
                        "TRT EP fp32 \nmean (ms)",
                        "TRT EP fp32 \n90th percentile (ms)",
                        "TRT EP fp32 \npeak memory usage (MiB)",
                        "Standalone TRT fp32 \nmean (ms)",
                        "Standalone TRT fp32 \n90th percentile (ms)",
                        "Standalone TRT fp32 \npeak memory usage (MiB)",
                        "TRT v CUDA EP fp32 \ngain (mean) (%)",
                        "EP v Standalone TRT fp32 \ngain (mean) (%)",
                        "CUDA fp16 \nmean (ms)",
                        "CUDA fp16 \n90th percentile (ms)",
                        "CUDA EP fp16 \npeak memory usage (MiB)",
                        "TRT EP fp16 \nmean (ms)",
                        "TRT EP fp16 \n90th percentile (ms)",
                        "TRT EP fp16 \npeak memory usage (MiB)",
                        "Standalone TRT fp16 \nmean (ms)",
                        "Standalone TRT fp16 \n90th percentile (ms)",
                        "Standalone TRT fp16 \npeak memory usage (MiB)",
                        "TRT v CUDA EP fp16 \ngain (mean) (%)", 
                        "EP v Standalone TRT fp16 \ngain (mean) (%)"]
        csv_writer = csv.writer(csv_file)

        if need_write_header:
            csv_writer.writerow(column_names)

        for key, value in results.items():
            cpu_average = "" 
            if cpu in value and "average_latency_ms" in value[cpu]:
                cpu_average = value[cpu]["average_latency_ms"]

            cpu_90_percentile = ""
            if cpu in value and "latency_90_percentile" in value[cpu]:
                cpu_90_percentile = value[cpu]["latency_90_percentile"]

            cuda_average = ""
            if cuda in value and 'average_latency_ms' in value[cuda]:
                cuda_average = value[cuda]['average_latency_ms']

            cuda_90_percentile = ""
            if cuda in value and 'latency_90_percentile' in value[cuda]:
                cuda_90_percentile = value[cuda]['latency_90_percentile']

            cuda_memory = ""
            if cuda in value and 'memory' in value[cuda]:
                cuda_memory = value[cuda]['memory']
            
            trt_average = ""
            if trt in value and 'average_latency_ms' in value[trt]:
                trt_average = value[trt]['average_latency_ms']

            trt_90_percentile = ""
            if trt in value and 'latency_90_percentile' in value[trt]:
                trt_90_percentile = value[trt]['latency_90_percentile']
            
            trt_memory = ""
            if trt in value and 'memory' in value[trt]:
                trt_memory = value[trt]['memory']

            standalone_trt_average = ""
            if standalone_trt in value and 'average_latency_ms' in value[standalone_trt]:
                standalone_trt_average = value[standalone_trt]['average_latency_ms']

            standalone_trt_90_percentile = ""
            if standalone_trt in value and 'latency_90_percentile' in value[standalone_trt]:
                standalone_trt_90_percentile = value[standalone_trt]['latency_90_percentile']
            
            standalone_trt_memory = ""
            if standalone_trt in value and 'memory' in value[standalone_trt]:
                standalone_trt_memory = value[standalone_trt]['memory']

            cuda_fp16_average = ""
            if cuda_fp16 in value and 'average_latency_ms' in value[cuda_fp16]:
                cuda_fp16_average = value[cuda_fp16]['average_latency_ms']

            cuda_fp16_memory = ""
            if cuda_fp16 in value and 'memory' in value[cuda_fp16]:
                cuda_fp16_memory = value[cuda_fp16]['memory']
            
            cuda_fp16_90_percentile = ""
            if cuda_fp16 in value and 'latency_90_percentile' in value[cuda_fp16]:
                cuda_fp16_90_percentile = value[cuda_fp16]['latency_90_percentile']

            trt_fp16_average = ""
            if trt_fp16 in value and 'average_latency_ms' in value[trt_fp16]:
                trt_fp16_average = value[trt_fp16]['average_latency_ms']

            trt_fp16_90_percentile = ""
            if trt_fp16 in value and 'latency_90_percentile' in value[trt_fp16]:
                trt_fp16_90_percentile = value[trt_fp16]['latency_90_percentile']

            trt_fp16_memory = ""
            if trt_fp16 in value and 'memory' in value[trt_fp16]:
                trt_fp16_memory = value[trt_fp16]['memory']
            
            standalone_trt_fp16_average = ""
            if standalone_trt_fp16 in value and 'average_latency_ms' in value[standalone_trt_fp16]:
                standalone_trt_fp16_average = value[standalone_trt_fp16]['average_latency_ms']

            standalone_trt_fp16_90_percentile = ""
            if standalone_trt_fp16 in value and 'latency_90_percentile' in value[standalone_trt_fp16]:
                standalone_trt_fp16_90_percentile = value[standalone_trt_fp16]['latency_90_percentile']
            
            standalone_trt_fp16_memory = ""
            if standalone_trt_fp16 in value and 'memory' in value[standalone_trt_fp16]:
                standalone_trt_fp16_memory = value[standalone_trt_fp16]['memory']

            row = [key,
                   cpu_average, 
                   cpu_90_percentile, 
                   cuda_average,
                   cuda_90_percentile,
                   cuda_memory,
                   trt_average,
                   trt_90_percentile,
                   trt_memory,
                   standalone_trt_average,
                   standalone_trt_90_percentile,
                   standalone_trt_memory,
                   value[trt_cuda_gain] if trt_cuda_gain in value else "  ",
                   value[trt_native_gain] if trt_native_gain in value else "  ",
                   cuda_fp16_average,
                   cuda_fp16_90_percentile,
                   cuda_fp16_memory,
                   trt_fp16_average,
                   trt_fp16_90_percentile,
                   trt_fp16_memory,
                   standalone_trt_fp16_average,
                   standalone_trt_fp16_90_percentile,
                   standalone_trt_fp16_memory,
                   value[trt_cuda_fp16_gain] if trt_cuda_fp16_gain in value else "  ",
                   value[trt_native_fp16_gain] if trt_native_fp16_gain in value else "  "
                   ]
            csv_writer.writerow(row)

    logger.info(f"CUDA/TRT latency comparison are saved to csv file: {csv_filename}")

def output_metrics(model_to_metrics, csv_filename):
    with open(csv_filename, mode="w", newline='') as csv_file:
        column_names = ["Model",
                        "% CUDA operators (not fall back to CPU)",
                        "Total TRT operators",
                        "Total operators",
                        "% TRT operator",
                        "Total TRT execution time",
                        "Total execution time",
                        "% TRT execution time"]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        results = []
        for model, ep_info in model_to_metrics.items():

            result = {}
            result_fp16 = {}
            result["model_name"] = model
            result_fp16["model_name"] = model + " (FP16)"

            if cuda in ep_info:
                result['ratio_of_ops_in_cuda_not_fallback_cpu'] = ep_info[cuda]['ratio_of_ops_in_cuda_not_fallback_cpu']

            if trt in ep_info:
                result['total_trt_execution_time'] = ep_info[trt]['total_trt_execution_time']
                result['total_execution_time'] = ep_info[trt]['total_execution_time']
                result['ratio_of_execution_time_in_trt'] = ep_info[trt]['ratio_of_execution_time_in_trt']

            if cuda in ep_info and trt in ep_info: 
                ########################################################################################
                # equation of % TRT ops:
                # (total ops in cuda json - cuda and cpu ops in trt json)/ total ops in cuda json
                ########################################################################################
                total_ops_in_cuda = ep_info[cuda]["total_ops"] 
                cuda_cpu_ops_in_trt = ep_info[trt]["total_ops"]

                result['total_ops_in_trt'] = total_ops_in_cuda - cuda_cpu_ops_in_trt
                result['total_ops'] = total_ops_in_cuda
                result['ratio_of_ops_in_trt'] = (total_ops_in_cuda - cuda_cpu_ops_in_trt) / total_ops_in_cuda

            if cuda_fp16 in ep_info:
                result_fp16['ratio_of_ops_in_cuda_not_fallback_cpu'] = ep_info[cuda_fp16]['ratio_of_ops_in_cuda_not_fallback_cpu']

            if trt_fp16 in ep_info:
                result_fp16['total_trt_execution_time'] = ep_info[trt_fp16]['total_trt_execution_time']
                result_fp16['total_execution_time'] = ep_info[trt_fp16]['total_execution_time']
                result_fp16['ratio_of_execution_time_in_trt'] = ep_info[trt_fp16]['ratio_of_execution_time_in_trt']

            if cuda_fp16 in ep_info and trt_fp16 in ep_info: 
                ########################################################################################
                # equation of % TRT ops:
                # (total ops in cuda json - cuda and cpu ops in trt json)/ total ops in cuda json
                ########################################################################################
                total_ops_in_cuda = ep_info[cuda_fp16]["total_ops"] 
                cuda_cpu_ops_in_trt = ep_info[trt_fp16]["total_ops"]

                result_fp16['total_ops_in_trt'] = total_ops_in_cuda - cuda_cpu_ops_in_trt
                result_fp16['total_ops'] = total_ops_in_cuda
                result_fp16['ratio_of_ops_in_trt'] = (total_ops_in_cuda - cuda_cpu_ops_in_trt) / total_ops_in_cuda

            
            results.append(result)
            results.append(result_fp16)


        
        for value in results:
            row = [value['model_name'],
                   value['ratio_of_ops_in_cuda_not_fallback_cpu'] if 'ratio_of_ops_in_cuda_not_fallback_cpu' in value else "  ",
                   value['total_ops_in_trt'] if 'total_ops_in_trt' in value else "  ",
                   value['total_ops'] if 'total_ops' in value else "  ",
                   value['ratio_of_ops_in_trt'] if 'ratio_of_ops_in_trt' in value else "  ",
                   value['total_trt_execution_time'] if 'total_trt_execution_time' in value else "  ",
                   value['total_execution_time'] if 'total_execution_time' in value else "  ",
                   value['ratio_of_execution_time_in_trt'] if 'ratio_of_execution_time_in_trt' in value else "  ",
                   ]
            csv_writer.writerow(row)

    logger.info(f"Tensorrt ratio metrics are saved to csv file: {csv_filename}")

def output_system_info(result, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "cpu_info", "cuda", "gpu_info", "linux_distro", "memory", "trt"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        csv_writer.writerow(result)

    logger.info(f"System information are saved to csv file: {csv_filename}")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--comparison", required=False, default="cuda_trt", choices=["cuda_trt", "acl"], help="EPs to compare: CPU vs. CUDA vs. TRT or CPU vs. ACL")

    parser.add_argument("-d", "--working_dir", required=False, default="./", help="Perf folder path with models")
    
    parser.add_argument("-m", "--model_source", required=False, default="model_list.json", help="Model source: (1) model list file (2) model directory.")

    parser.add_argument("-r", "--running_mode", required=False, default="benchmark", choices=["validate", "benchmark"], help="Testing mode.")

    parser.add_argument("-i", "--input_data", required=False, default="fix", choices=["fix", "random"], help="Type of input data.")

    parser.add_argument("-o", "--perf_result_path", required=False, default="result", help="Directory for perf result.")
    
    parser.add_argument("-w", "--workspace", required=False, default="/", help="Workspace to find tensorrt")
    
    parser.add_argument("--track_memory", required=False, default=True, help="Track CUDA and TRT Memory Usage")

    parser.add_argument("--ep", required=False, default=None, help="Specify ORT Execution Provider.")
    
    parser.add_argument("--ep_list", nargs="+", required=False, default=None, help="Specify ORT Execution Providers list.")

    parser.add_argument("--fp16", required=False, default=True, action="store_true", help="Inlcude Float16 into benchmarking.")

    parser.add_argument("--trtexec", required=False, default=None, help="trtexec executable path.")

    # Validation options
    parser.add_argument("--percent_mismatch", required=False, default=20.0, help="Allowed percentage of mismatched elements in validation.")
    parser.add_argument("--rtol", required=False, default=0, help="Relative tolerance for validating outputs.")
    parser.add_argument("--atol", required=False, default=20, help="Absolute tolerance for validating outputs.")
    
    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=1,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

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
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(message)s')
        logging.getLogger("transformers").setLevel(logging.WARNING)

def parse_models_helper(args, models): 
    if ".json" in args.model_source:
        logger.info("Parsing model information from file ...")
        parse_models_info_from_file(args.working_dir, args.model_source, models)
    else:
        logger.info("Parsing model information from directory ...")
        parse_models_info_from_directory(args.model_source, models)

def main():
    args = parse_arguments()
    setup_logger(False)
    pp = pprint.PrettyPrinter(indent=4)
    
    logger.info("\n\nStart perf run ...\n")

    models = {}
    parse_models_helper(args, models)

    if not os.path.exists("symbolic_shape_infer.py"):
        p1 = subprocess.Popen(["sudo", "wget", "https://raw.githubusercontent.com/microsoft/onnxruntime/master/onnxruntime/python/tools/symbolic_shape_infer.py"])
        p1.wait()
    os.environ["SYMBOLIC_SHAPE_INFER"] = os.path.join(os.getcwd(), "symbolic_shape_infer.py")

    perf_start_time = datetime.now()
    success_results, model_to_latency, model_to_fail_ep, model_to_metrics = run_onnxruntime(args, models)
    perf_end_time = datetime.now()

    logger.info("Done running the perf.")
    logger.info("\nTotal time for benchmarking all models: {}".format(perf_end_time - perf_start_time))
    logger.info(list(models.keys()))

    logger.info("\nTotal models: {}".format(len(models)))
    
    fail_model_cnt = 0
    for key, value in models.items():
        if key in model_to_fail_ep: fail_model_cnt += 1
    logger.info("Fail models: {}".format(fail_model_cnt))
    logger.info("Success models: {}".format(len(models) - fail_model_cnt ))

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
        pp.pprint(model_to_latency)
        write_map_to_file(model_to_latency, LATENCY_FILE)
        if args.write_test_result:
            csv_filename = args.benchmark_latency_csv if args.benchmark_latency_csv else f"benchmark_latency_{time_stamp}.csv"
            csv_filename = os.path.join(path, csv_filename)
            output_latency(model_to_latency, csv_filename)
    
    if success_results:
        csv_filename = args.benchmark_success_csv if args.benchmark_success_csv else f"benchmark_success_{time_stamp}.csv"
        csv_filename = os.path.join(path, csv_filename)
        output_details(success_results, csv_filename)

    if len(model_to_metrics) > 0:
        logger.info("\n=========================================")
        logger.info("========== Models/EPs metrics  ==========")
        logger.info("=========================================")
        pp.pprint(model_to_metrics)
        write_map_to_file(model_to_metrics, METRICS_FILE)

        if args.write_test_result:
            csv_filename = args.benchmark_metrics_csv if args.benchmark_metrics_csv else f"benchmark_metrics_{time_stamp}.csv"
            csv_filename = os.path.join(path, csv_filename)
            output_metrics(model_to_metrics, csv_filename)

if __name__ == "__main__":
    main()
