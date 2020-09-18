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
from float16 import *
# import torch

debug = False
sys.path.append('.')
logger = logging.getLogger('')

ep_to_provider_list = {
    "CPUExecutionProvider": ["CPUExecutionProvider"],
    "CUDAExecutionProvider": ["CUDAExecutionProvider"],
    "CUDAExecutionProvider_fp16": ["CUDAExecutionProvider"],
    "TensorrtExecutionProvider": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    "TensorrtExecutionProvider_fp16": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
}

def run_trt_standalone(trtexec, model_path, ort_inputs, all_inputs_shape, fp16):
    model_path = "--onnx=" + model_path
    input_shape = []

    print(all_inputs_shape)

    for i in range(len(ort_inputs)):
        name = ort_inputs[i].name

        shape = []
        for j in all_inputs_shape[i]:
            shape.append(str(j))
        shape = "x".join(shape)
        shape = name + ':' + shape
        input_shape.append(shape)

    shapes_arg = '--optShapes=' + ','.join(input_shape)
    print(shapes_arg)

    result = {}
    try:

        if fp16:
            p1 = subprocess.Popen([trtexec, model_path, "--fp16", "--percentile=90", "--explicitBatch", shapes_arg], stdout=subprocess.PIPE)
        else:
            p1 = subprocess.Popen([trtexec, model_path, "--percentile=90", "--explicitBatch", shapes_arg], stdout=subprocess.PIPE)
        stdout, sterr = p1.communicate()
        print(stdout)
        stdout = stdout.decode("ascii").strip()

        tmp = stdout.split("\n")
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

        print(result)
        return result

    except Exception as e:
        logger.info("trtexec fails...")
        return None



def get_latency_result(runtimes, batch_size):
    latency_ms = sum(runtimes) / float(len(runtimes)) * 1000.0
    latency_variance = numpy.var(runtimes, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    return {
        "test_times": len(runtimes),
        "latency_variance": "{:.2f}".format(latency_variance),
        "latency_90_percentile": "{:.2f}".format(numpy.percentile(runtimes, 90) * 1000.0),
        "latency_95_percentile": "{:.2f}".format(numpy.percentile(runtimes, 95) * 1000.0),
        "latency_99_percentile": "{:.2f}".format(numpy.percentile(runtimes, 99) * 1000.0),
        "average_latency_ms": "{:.2f}".format(latency_ms),
        "QPS": "{:.2f}".format(throughput),
    }


def get_ort_session_inputs_and_outptus(name, session, ort_input):

    sess_inputs = {}
    sess_outputs = None

    if name == 'BERT-Squad':
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

    elif name == 'BiDAF':
        sess_inputs = {
                "context_word": ort_input[0],
                "context_char": ort_input[2],
                "query_word": ort_input[1],
                "query_char": ort_input[3]}
        sess_outputs = ["start_pos","end_pos"]

    elif name == 'Yolov4':
        sess_inputs[session.get_inputs()[0].name] = ort_input[0]
        sess_outputs = ['Identity:0']

    elif name == 'Shufflenet-v2':
        sess_inputs[session.get_inputs()[0].name] = ort_input

    else:
        sess_inputs = {}
        for i in range(len(session.get_inputs())):
            sess_inputs[session.get_inputs()[i].name] = ort_input[i]

    return (sess_inputs, sess_outputs)

def inference_ort(args, name, session, ep, ort_inputs, result_template, repeat_times, batch_size):

    runtimes = []
    for ort_input in ort_inputs:
        sess_inputs, sess_outputs = get_ort_session_inputs_and_outptus(name, session, ort_input)
        print("sess_inputs:")
        print(sess_inputs)
        print("sess_outputs:")
        print(sess_outputs)
        try:
            if args.input_data == "random":
                repeat_times = 1 # warn-up run is included in ort_inputs
            else:
                repeat_times += 1 # add warn-up run

            runtime = timeit.repeat(lambda: session.run(sess_outputs, sess_inputs), number=1, repeat=repeat_times)
            runtimes += runtime

        except Exception as e:
            logger.error(e)
            return None

    print(runtimes)
    runtimes[:] = runtimes[1:]
    print(runtimes)

    result = {}
    result.update(result_template)
    result.update({"io_binding": False})
    result.update(get_latency_result(runtimes, batch_size))
    return result

def inference_ort_and_get_prediction(name, session, ort_inputs):

    ort_outputs = []
    for ort_input in ort_inputs:
        sess_inputs, sess_outputs = get_ort_session_inputs_and_outptus(name, session, ort_input)
        print("sess_inputs:")
        print(sess_inputs)
        print("sess_outputs:")
        print(sess_outputs)
        try:
            result = session.run(sess_outputs, sess_inputs)

            # handle shape of output differently
            if name == 'BERT-Squad':
                ort_outputs.append([result])
            elif name == 'Shufflenet-v2':
                ort_outputs.append(result[0])
            else:
                ort_outputs.append(result)
        except Exception as e:
            logger.error(e)
            return None

    return ort_outputs

# not use for this script yet
def inference_ort_with_io_binding(model, ort_inputs, result_template, repeat_times, batch_size, device='cuda'):
    runtimes = []

    session = model.get_session()

    # Bind inputs and outputs to onnxruntime session
    io_binding = session.io_binding()

    for ort_input in ort_inputs:

        # Bind inputs to device
        if model.get_model_name() == 'BERT-Squad':
            name = session.get_inputs()[0].name
            print(name)
            np_input = torch.from_numpy(ort_input[0]).to(device)
            io_binding.bind_input(name, np_input.device.type, 0, numpy.longlong, np_input.shape, np_input.data_ptr())
            name = session.get_inputs()[1].name
            print(name)
            np_input = torch.from_numpy(ort_input[1][0:1]).to(device)
            io_binding.bind_input(name, np_input.device.type, 0, numpy.longlong, np_input.shape, np_input.data_ptr())
            name = session.get_inputs()[2].name
            print(name)
            np_input = torch.from_numpy(ort_input[2][0:1]).to(device)
            io_binding.bind_input(name, np_input.device.type, 0, numpy.longlong, np_input.shape, np_input.data_ptr())
            name = session.get_inputs()[3].name
            print(name)
            np_input = torch.from_numpy(ort_input[3][0:1]).to(device)
            io_binding.bind_input(name, np_input.device.type, 0, numpy.longlong, np_input.shape, np_input.data_ptr())
        else:
            name = session.get_inputs()[0].name
            print(ort_input[0])
            np_input = torch.from_numpy(ort_input[0]).to(device)
            io_binding.bind_input(name, np_input.device.type, 0, numpy.float32, np_input.shape, np_input.data_ptr())

        name_o = session.get_outputs()[0].name
        io_binding.bind_output(name_o)

        # name = session.get_inputs()[0].name
        # np_input = torch.from_numpy(numpy.asarray(ort_inputs[0][0])).to(device)
        # io_binding.bind_input(name, np_input.device.type, 0, numpy.float32, np_input.shape, np_input.data_ptr())
        # name_o = session.get_outputs()[0].name
        # io_binding.bind_output(name_o, 'cpu', 0, numpy.float32, session.get_outputs()[0].shape, None)

        try:
            runtimes = runtimes + timeit.repeat(lambda: session.run_with_iobinding(io_binding), number=1, repeat=repeat_times)
        except Exception as e:
            logger.error(e)
            return None

    print(runtimes)

    result = {}
    result.update(result_template)
    result.update({"io_binding": True})
    result.update(get_latency_result(runtimes, batch_size))
    return result


def get_cuda_version():
    from pathlib import Path
    home = str(Path.home())

    p1 = subprocess.Popen(["find", home+"/.local/lib/", "-name", "onnxruntime_pybind11_state.so"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("ascii").strip()
    p1 = subprocess.Popen(["ldd", stdout], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "libcudart.so"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()

    return stdout

def get_trt_version():
    from pathlib import Path
    home = str(Path.home())

    p1 = subprocess.Popen(["find", home+"/.local/lib/", "-name", "onnxruntime_pybind11_state.so"], stdout=subprocess.PIPE)
    stdout, sterr = p1.communicate()
    stdout = stdout.decode("ascii").strip()
    p1 = subprocess.Popen(["ldd", stdout], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "libnvinfer.so"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()

    if stdout == "":
        p1 = subprocess.Popen(["find", home+"/.local/lib/", "-name", "libonnxruntime_providers_tensorrt.so"], stdout=subprocess.PIPE)
        stdout, sterr = p1.communicate()
        stdout = stdout.decode("ascii").strip()
        p1 = subprocess.Popen(["ldd", stdout], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep", "libnvinfer.so"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()

    return stdout

# not use for this script temporarily
def tmp_get_trt_version():
    p1 = subprocess.Popen(["dpkg", "-l"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "TensorRT runtime libraries"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()

    if stdout != "":
        stdout = re.sub('\s+', ' ', stdout)
        return stdout

    if os.path.exists("/usr/lib/x86_64-linux-gnu/libnvinfer.so"):
        p1 = subprocess.Popen(["readelf", "-s", "/usr/lib/x86_64-linux-gnu/libnvinfer.so"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep", "version"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()
        stdout = stdout.split(" ")[-1]
        return stdout

    elif os.path.exists("/usr/lib/aarch64-linux-gnu/libnvinfer.so"):
        p1 = subprocess.Popen(["readelf", "-s", "/usr/lib/aarch64-linux-gnu/libnvinfer.so"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep", "version"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()
        stdout = stdout.split(" ")[-1]
        return stdout

    return ""

#
# The following two lists will be generated.
#
# inputs: [[test_data_0_input_0.pb, test_data_0_input_1.pb ...], [test_data_1_input_0.pb, test_data_1_input_1.pb ...] ...]
# outputs: [[test_data_0_output_0.pb, test_data_0_output_1.pb ...], [test_data_1_output_0.pb, test_data_1_output_1.pb ...] ...]
#
def load_onnx_model_zoo_test_data(path, all_inputs_shape, data_type="fp32"):
    print("Parsing test data in {} ...".format(path))
    # p1 = subprocess.Popen(["find", path, "-name", "test_data_set*", "-type", "d"], stdout=subprocess.PIPE)
    p1 = subprocess.Popen(["find", path, "-name", "test_data*", "-type", "d"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    test_data_set_dir = stdout.split("\n")
    print(test_data_set_dir)

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
        print(input_data)

        input_data_pb = []
        for data in input_data:
            tensor = onnx.TensorProto()
            with open(data, 'rb') as f:
                tensor.ParseFromString(f.read())

                tensor_to_array = numpy_helper.to_array(tensor)

                if data_type == "fp16" and tensor_to_array.dtype == np.dtype(np.float32):
                    tensor_to_array = tensor_to_array.astype(np.float16)
                input_data_pb.append(tensor_to_array)

                # print(np.array(input_data_pb[-1]).shape)
                if not shape_flag:
                    all_inputs_shape.append(input_data_pb[-1].shape)
                print(all_inputs_shape[-1])
        inputs.append(input_data_pb)
        print('Loaded {} inputs successfully.'.format(len(inputs)))

        # load outputs
        p1 = subprocess.Popen(["find", ".", "-name", "output*"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()
        output_data = stdout.split("\n")
        print(output_data)

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

                    print(np.array(output_data_pb[-1]).shape)
            outputs.append(output_data_pb)
            print('Loaded {} outputs successfully.'.format(len(outputs)))

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
            print("original tensor:")
            print(tensor)
            print("new random tensor:")
            print(new_tensor)
            print("\n")
            input_data.append(new_tensor)
        inputs.append(input_data)

    return inputs

def validate(all_ref_outputs, all_outputs, decimal):
    print('Reference {} results.'.format(len(all_ref_outputs)))
    print('Predicted {} results.'.format(len(all_outputs)))
    print('decimal {}'.format(decimal))
    # print(np.array(all_ref_outputs).shape)
    # print(np.array(all_outputs).shape)

    try:
        for i in range(len(all_outputs)):
            ref_outputs = all_ref_outputs[i]
            outputs = all_outputs[i]

            for j in range(len(outputs)):
                ref_output = ref_outputs[j]
                output = outputs[j]
                # print(ref_output)
                # print(output)

                # Compare the results with reference outputs up to x decimal places
                for ref_o, o in zip(ref_output, output):
                    # abs(desired-actual) < 1.5 * 10**(-decimal)
                    np.testing.assert_almost_equal(ref_o, o, decimal)
    except Exception as e:
        logger.error(e)
        return False, e

    print('ONNX Runtime outputs are similar to reference outputs!')
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
            print(f)
            continue
        subprocess.Popen(["rm","-rf", f], stdout=subprocess.PIPE)

def remove_profiling_files(path):
    files = []
    p = subprocess.Popen(["find", path, "-name", "onnxruntime_profile*"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    files = files + stdout.split("\n")

    for f in files:
        if "custom_test_data" in f:
            continue
        subprocess.Popen(["rm","-rf", f], stdout=subprocess.PIPE)


def update_fail_report(fail_results, args, model, ep, e_type, e):
    result = {}

    result["model"] = model
    result["ep"] = ep
    result["error type"] = e_type
    result["error message"] = re.sub('^\n', '', str(e))

    fail_results.append(result)


def update_fail_model(model_ep_fail_map, fail_results, args, model_name, ep, e_type, e):

    if not model_name in model_ep_fail_map:
        model_ep_fail_map[model_name] = [ep]
    else:
        if ep not in model_ep_fail_map[model_name]:
            model_ep_fail_map[model_name].append(ep)

    update_fail_report(fail_results, args, model_name, ep, e_type, e)

    # If TRT fails, TRT FP16 should fail as well
    if ep == 'TensorrtExecutionProvider':
        ep_ = "TensorrtExecutionProvider_fp16"
        e_ = "Not benchmarking TRT FP16 since TRT failed already."
        update_fail_report(fail_results, args, model_name, ep_, e_type, e_)
        model_ep_fail_map[model_name].append(ep_)


def skip_ep(model_name, ep, model_ep_fail_map):
    if model_name == 'vision-yolov3' and "fp16" in ep:
        return True

    if model_name == 'speech' and "fp16" in ep:
        return True

    if model_name not in model_ep_fail_map:
        return False

    ep_fail_list = model_ep_fail_map[model_name]

    if ep in ep_fail_list:
        return True

    return False

def read_model_ep_fail_map_from_file(map_file):
    with open(map_file) as f:
        try:
            data = json.load(f)
        except Exception as e:
            return None

    return data

def write_model_ep_fail_map_to_file(model_ep_fail_map):
    with open('.model_ep_fail_map.json', 'w') as file:
     file.write(json.dumps(model_ep_fail_map)) # use `json.loads` to do the reverse

def get_system_info(info):
    info["cuda"] = get_cuda_version()
    info["trt"] = get_trt_version()

    p = subprocess.Popen(["cat", "/etc/os-release"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    stdout = stdout.split("\n")[:2]
    infos = []
    for row in stdout:
        row = re.sub('=', ':  ', row)
        row = re.sub('"', '', row)
        infos.append(row)
    info["linux_distro"] = infos

    p = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    stdout = stdout.split("\n")
    infos = []
    for row in stdout:
        if "mode" in row or "Arch" in row or "name" in row:
            # row = row.replace(":\s+", ":  ")
            row = re.sub(': +', ':  ', row)
            infos.append(row)
    info["cpu_info"] = infos

    p1 = subprocess.Popen(["lspci", "-v"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "NVIDIA"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    stdout = stdout.split("\n")
    infos = []
    for row in stdout:
        row = re.sub('.*:', '', row)
        infos.append(row)
    info["gpu_info"] = infos

    p = subprocess.Popen(["cat", "/proc/meminfo"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    stdout = stdout.split("\n")
    infos = []
    for row in stdout:
        if "Mem" in row:
            row = re.sub(': +', ':  ', row)
            infos.append(row)
    info["memory"] = infos

def parse_models_info(path):
    models = {}

    with open(path) as f:
        data = json.load(f)

        for row in data:

            if 'model_name' in row:
                models[row['model_name']] = {}
            else:
                logger.error('Model name must be provided in models_info.json')
                raise

            model = models[row['model_name']]

            if 'working_directory' in row:
                model['working_directory'] = row['working_directory']
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

    return models

def convert_model_from_float_to_float16(model_path):
    # from onnxmltools.utils.float16_converter import convert_float_to_float16
    from onnxmltools.utils import load_model, save_model
    from float16 import convert_float_to_float16

    onnx_model = load_model(model_path)
    new_onnx_model = convert_float_to_float16(onnx_model)
    save_model(new_onnx_model, 'new_fp16_model.onnx')

    return os.path.join(os.getcwd(), "new_fp16_model.onnx")

def create_session(model_path, providers, session_options):
    logger.info(model_path)
    try:
        session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=session_options)
        return session
    except:
        logger.info("Use symbolic_shape_infer.py")

    try:
        new_model_path = model_path[:].replace(".onnx", "_new.onnx")

        if not os.path.exists(new_model_path):
            subprocess.run("python3 -m onnxruntime.tools.symbolic_shape_infer --input " + model_path + " --output " + new_model_path + " --auto_merge", shell=True, check=True)
        session = onnxruntime.InferenceSession(new_model_path, providers=providers, sess_options=session_options)
        return session
    except Exception as e:
        print(e)
        raise

def run_onnxruntime(args, models):

    success_results = []
    fail_results = []
    latency_comparison_map = {} # model -> CUDA/TRT latency
    profile_metrics_map = {} # model -> metrics from profiling file
    model_ep_fail_map = {} # model -> failing ep


    # read failing ep information if file exists
    if args.running_mode == 'benchmark':
        if os.path.exists('.model_ep_fail_map.json'):
            model_ep_fail_map = read_model_ep_fail_map_from_file('.model_ep_fail_map.json')

    if args.fp16:
        ep_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CUDAExecutionProvider_fp16", "TensorrtExecutionProvider_fp16"]
    else:
        ep_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]

    validation_exemption = ["TensorrtExecutionProvider_fp16"]

    #######################
    # iterate model
    #######################
    for name, info in models.items():
        latency_result = {}
        path = info["working_directory"]

        pwd = os.getcwd()
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        path = os.getcwd()

        # cleanup files before running a new inference
        if args.running_mode == "validate":
            remove_profiling_files(path)

        inputs = []
        ref_outputs = []
        inputs_fp32 = []
        ref_outputs_fp32 = []
        inputs_fp16 = []
        ref_outputs_fp16 = []
        all_inputs_shape = [] # use for standalone trt
        ep_to_ep_op_map = {} # ep -> { ep -> operator }
        profile_already_parsed = set()


        #######################
        # iterate ep
        #######################
        for ep in ep_list:

            if skip_ep(name, ep, model_ep_fail_map):
                continue

            ep_ = ep_to_provider_list[ep][0]
            if (ep_ not in onnxruntime.get_available_providers()):
                logger.error("No {} support".format(ep_))
                continue

            model_path = info["model_path"]

            if "fp16" in ep:
                fp16 = True
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

                if ep == "CUDAExecutionProvider_fp16":
                    model_path = convert_model_from_float_to_float16(model_path)

                logger.info("\nInitializing {} with float16 enabled to run on {} ...".format(name, ep))
            else:
                fp16 = False
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"
                logger.info("\nInitializing {} to run on {} ...".format(name, ep))


            test_data_dir = info["test_data_path"]

            # read input/output of test data
            if fp16 and ep == "CUDAExecutionProvider_fp16":
                if not inputs_fp16 or not ref_outputs_fp16:
                    inputs_fp16, ref_outputs_fp16 = load_onnx_model_zoo_test_data(test_data_dir, all_inputs_shape, "fp16")

                inputs = inputs_fp16
                ref_outputs = ref_outputs_fp16
            else:
                if not inputs_fp32 or not ref_outputs_fp32:
                    inputs_fp32, ref_outputs_fp32 = load_onnx_model_zoo_test_data(test_data_dir, all_inputs_shape)

                inputs = inputs_fp32
                ref_outputs = ref_outputs_fp32


            if args.input_data == "random":
                inputs = generate_onnx_model_random_input(args.test_times+1, inputs[0])

            #######################################
            # benchmark or validation
            #######################################
            if args.running_mode == 'benchmark':
                logger.info("===========================")
                logger.info("======== benchmark ========")
                logger.info("===========================")

                options = onnxruntime.SessionOptions()
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

                # create onnxruntime inference session
                try:
                    sess = create_session(model_path, ep_to_provider_list[ep], options)

                except Exception as e:
                    logger.error(e)
                    # update_fail_model(model_ep_fail_map, fail_results, args, name, ep, e)
                    continue

                logger.info("[start] Begin to inference {} with {} ...".format(name, ep))
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

                result = inference_ort(args, name, sess, ep, inputs, result_template, args.test_times, batch_size)
                if result:
                    success_results.append(result)
                    logger.info(result)

                    latency_result[ep] = {}
                    latency_result[ep]["average_latency_ms"] = result["average_latency_ms"]
                    latency_result[ep]["latency_90_percentile"] = result["latency_90_percentile"]

                    # get standalone TensorRT perf
                    if "TensorrtExecutionProvider" in ep and args.trtexec:
                        result = run_trt_standalone(args.trtexec, model_path, sess.get_inputs(), all_inputs_shape, fp16)
                        if result and len(result) > 0:
                            if fp16:
                                latency_result["Standalone_TRT_fp16"] = result
                            else:
                                latency_result["Standalone_TRT"] = result

                    latency_comparison_map[name] = copy.deepcopy(latency_result)



            elif args.running_mode == 'validate':
                logger.info("==========================")
                logger.info("======== validate ========")
                logger.info("==========================")

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
                    update_fail_model(model_ep_fail_map, fail_results, args, name, ep, 'runtime error', e)
                    continue

                sess.disable_fallback()

                logger.info("Start to inference {} with {} ...".format(name, ep))
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

                        decimal = 0

                        status = validate(ref_outputs, ort_outputs, decimal)
                        if not status[0]:
                            update_fail_model(model_ep_fail_map, fail_results, args, name, ep, 'result accuracy issue', status[1])
                            continue
                    except Exception as e:
                        logger.error(e)
                        update_fail_model(model_ep_fail_map, fail_results, args, name, ep, 'runtime error', e)
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
                metrics = get_profile_metrics(path, profile_already_parsed)
                if metrics:
                    print(ep)
                    ep_to_ep_op_map[ep] = metrics

        ####################
        # end of iterate ep
        ####################


        # get percentage of execution time and operators in TRT
        if len(ep_to_ep_op_map) > 0:
            trt_op_map = None
            trt_fp16_op_map = None
            cuda_op_map = None
            cuda_fp16_op_map = None

            for ep, op_map in ep_to_ep_op_map.items():
                if ep == "CUDAExecutionProvider":
                    cuda_op_map = op_map
                elif ep == "CUDAExecutionProvider_fp16":
                    cuda_fp16_op_map = op_map
                elif ep == "TensorrtExecutionProvider":
                    trt_op_map = op_map
                elif ep == "TensorrtExecutionProvider_fp16":
                    trt_fp16_op_map = op_map

            profile_metrics_map[name] = {}

            if cuda_op_map:
                profile_metrics_map[name]['ratio_of_ops_in_cuda_not_fallback_cpu'] = calculate_cuda_op_percentage(cuda_op_map) 

            if trt_op_map:
                total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_op_map)
                profile_metrics_map[name]['total_trt_execution_time'] = total_trt_execution_time
                profile_metrics_map[name]['total_execution_time'] = total_execution_time
                profile_metrics_map[name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt
                if cuda_op_map:
                    total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_op_map, cuda_op_map)
                    profile_metrics_map[name]['total_ops_in_trt'] = total_ops_in_trt
                    profile_metrics_map[name]['total_ops'] = total_ops
                    profile_metrics_map[name]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt

            if trt_fp16_op_map:
                total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_fp16_op_map)
                name_ = name + " (FP16)"
                profile_metrics_map[name_] = {}
                profile_metrics_map[name_]['total_trt_execution_time'] = total_trt_execution_time
                profile_metrics_map[name_]['total_execution_time'] = total_execution_time
                profile_metrics_map[name_]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt
                if cuda_fp16_op_map:
                    total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_fp16_op_map, cuda_op_map)
                    profile_metrics_map[name_]['total_ops_in_trt'] = total_ops_in_trt
                    profile_metrics_map[name_]['total_ops'] = total_ops
                    profile_metrics_map[name_]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt

            if debug:
                pp = pprint.PrettyPrinter(indent=4)
                print('CUDA operator map:')
                pp.pprint(cuda_op_map)
                print('TRT operator map:')
                pp.pprint(trt_op_map)
                print('CUDA FP16 operator map:')
                pp.pprint(cuda_fp16_op_map)
                print('TRT FP16 operator map:')
                pp.pprint(trt_fp16_op_map)

        # cleanup_files()
        os.chdir(pwd)

        # end of model

    return success_results, fail_results, latency_comparison_map, model_ep_fail_map, profile_metrics_map

def add_improvement_information(latency_comparison_map):
    for key, value in latency_comparison_map.items():
        if not ('TensorrtExecutionProvider' in value and 'CUDAExecutionProvider' in value):
            continue

        trt_latency = float(value['TensorrtExecutionProvider']['average_latency_ms'])
        cuda_latency = float(value['CUDAExecutionProvider']['average_latency_ms'])
        gain = (cuda_latency - trt_latency)*100/cuda_latency
        value["Tensorrt_gain(%)"] = "{:.2f} %".format(gain)

        if "TensorrtExecutionProvider_fp16" in value and "CUDAExecutionProvider_fp16" in value:
            trt_fp16_latency = float(value['TensorrtExecutionProvider_fp16']['average_latency_ms'])
            cuda_fp16_latency = float(value['CUDAExecutionProvider_fp16']['average_latency_ms'])
            gain = (cuda_fp16_latency - trt_fp16_latency)*100/cuda_fp16_latency
            value["Tensorrt_fp16_gain(%)"] = "{:.2f} %".format(gain)

def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "fp16", "io_binding", "model_name", "inputs", "batch_size",
            "sequence_length", "datetime", "test_times", "QPS", "average_latency_ms", "latency_variance",
            "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Detail results are saved to csv file: {csv_filename}")

def output_fail(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "model", "ep", "error type", "error message"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Failing results are saved to csv file: {csv_filename}")

def output_latency(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["Model",
                        "CUDA \nmean (ms)",
                        "CUDA \n90th percentile (ms)",
                        "TRT EP \nmean (ms)",
                        "TRT EP \n90th percentile (ms)",
                        "Standalone TRT \nmean (ms)",
                        "Standalone TRT \n90th percentile (ms)",
                        "CUDA fp16 \nmean (ms)",
                        "CUDA fp16 \n90th percentile (ms)",
                        "TRT EP fp16 \nmean (ms)",
                        "TRT EP fp16 \n90 percentile (ms)",
                        "Standalone TRT fp16 \nmean (ms)",
                        "Standalone TRT fp16 \n90th percentile (ms)",
                        "TRT EP \ngain (mean) (%)",
                        "TRT EP fp16 \ngain (mean) (%)"]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        for key, value in results.items():
            cuda_average = ""
            if 'CUDAExecutionProvider' in value and 'average_latency_ms' in value['CUDAExecutionProvider']:
                cuda_average = value['CUDAExecutionProvider']['average_latency_ms']

            cuda_99_percentile = ""
            if 'CUDAExecutionProvider' in value and 'latency_90_percentile' in value['CUDAExecutionProvider']:
                cuda_99_percentile = value['CUDAExecutionProvider']['latency_90_percentile']

            trt_average = ""
            if 'TensorrtExecutionProvider' in value and 'average_latency_ms' in value['TensorrtExecutionProvider']:
                trt_average = value['TensorrtExecutionProvider']['average_latency_ms']

            trt_99_percentile = ""
            if 'TensorrtExecutionProvider' in value and 'latency_90_percentile' in value['TensorrtExecutionProvider']:
                trt_99_percentile = value['TensorrtExecutionProvider']['latency_90_percentile']

            standalone_trt_average = ""
            if 'Standalone_TRT' in value and 'average_latency_ms' in value['Standalone_TRT']:
                standalone_trt_average = value['Standalone_TRT']['average_latency_ms']

            standalone_trt_99_percentile = ""
            if 'Standalone_TRT' in value and 'latency_90_percentile' in value['Standalone_TRT']:
                standalone_trt_99_percentile = value['Standalone_TRT']['latency_90_percentile']


            cuda_fp16_average = ""
            if 'CUDAExecutionProvider_fp16' in value and 'average_latency_ms' in value['CUDAExecutionProvider_fp16']:
                cuda_fp16_average = value['CUDAExecutionProvider_fp16']['average_latency_ms']

            cuda_fp16_99_percentile = ""
            if 'CUDAExecutionProvider_fp16' in value and 'latency_90_percentile' in value['CUDAExecutionProvider_fp16']:
                cuda_fp16_99_percentile = value['CUDAExecutionProvider_fp16']['latency_90_percentile']

            trt_fp16_average = ""
            if 'TensorrtExecutionProvider_fp16' in value and 'average_latency_ms' in value['TensorrtExecutionProvider_fp16']:
                trt_fp16_average = value['TensorrtExecutionProvider_fp16']['average_latency_ms']

            trt_fp16_99_percentile = ""
            if 'TensorrtExecutionProvider_fp16' in value and 'latency_90_percentile' in value['TensorrtExecutionProvider_fp16']:
                trt_fp16_99_percentile = value['TensorrtExecutionProvider_fp16']['latency_90_percentile']

            standalone_trt_fp16_average = ""
            if 'Standalone_TRT_fp16' in value and 'average_latency_ms' in value['Standalone_TRT_fp16']:
                standalone_trt_fp16_average = value['Standalone_TRT_fp16']['average_latency_ms']

            standalone_trt_fp16_99_percentile = ""
            if 'Standalone_TRT_fp16' in value and 'latency_90_percentile' in value['Standalone_TRT_fp16']:
                standalone_trt_fp16_99_percentile = value['Standalone_TRT_fp16']['latency_90_percentile']


            row = [key,
                   cuda_average,
                   cuda_99_percentile,
                   trt_average,
                   trt_99_percentile,
                   standalone_trt_average,
                   standalone_trt_99_percentile,
                   cuda_fp16_average,
                   cuda_fp16_99_percentile,
                   trt_fp16_average,
                   trt_fp16_99_percentile,
                   standalone_trt_fp16_average,
                   standalone_trt_fp16_99_percentile,
                   value['Tensorrt_gain(%)'] if 'Tensorrt_gain(%)' in value else "  ",
                   value['Tensorrt_fp16_gain(%)'] if 'Tensorrt_fp16_gain(%)' in value else "  "
                   ]
            csv_writer.writerow(row)

    logger.info(f"CUDA/TRT latency comparison are saved to csv file: {csv_filename}")

def output_ratio(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
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

        for key, value in results.items():
            row = [key,
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_list_file", required=False, default="model_list.json", help="Model list file.")

    parser.add_argument("-r", "--running_mode", required=False, default="benchmark", choices=["validate", "benchmark"], help="Testing mode.")

    parser.add_argument("-i", "--input_data", required=False, default="zoo", choices=["zoo", "random"], help="source of input data.")

    parser.add_argument("--fp16", required=False, default=True, action="store_true", help="Inlcude Float16 into benchmarking.")

    parser.add_argument("--trtexec", required=False, default=None, help="trtexec executable path.")

    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=1,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

    args = parser.parse_args()
    return args

def setup_logger(verbose):
    if verbose:
        coloredlogs.install(level='DEBUG', fmt='[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')
    else:
        coloredlogs.install(fmt='%(message)s')
        logging.getLogger("transformers").setLevel(logging.WARNING)

def main():
    args = parse_arguments()
    setup_logger(False)
    pp = pprint.PrettyPrinter(indent=4)

    models = parse_models_info(args.model_list_file)

    perf_start_time = datetime.now()
    success_results, fail_results, latency_comparison_map, failing_models, profile_metrics_map = run_onnxruntime(args, models)
    perf_end_time = datetime.now()

    logger.info("\nTotal time for running/profiling all models: {}".format(perf_end_time - perf_start_time))
    logger.info(list(models.keys()))

    logger.info("\nTotal models: {}".format(len(models)))
    logger.info("Fail models: {}".format(len(failing_models)))
    logger.info("Models FAIL/SUCCESS: {}/{}".format(len(failing_models), len(models) - len(failing_models)))

    path = "result"
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        os.mkdir(path)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if len(failing_models) > 0:
        logger.info("\n============================================")
        logger.info("========== Failing Models/EPs ==============")
        logger.info("============================================")
        logger.info(failing_models)
        write_model_ep_fail_map_to_file(failing_models)

    if latency_comparison_map:
        logger.info("\n=========================================")
        logger.info("=========== CUDA/TRT latency  ===========")
        logger.info("=========================================")
        add_improvement_information(latency_comparison_map)
        pp.pprint(latency_comparison_map)
        csv_filename = f"benchmark_latency_{time_stamp}.csv"
        csv_filename = os.path.join(path, csv_filename)
        output_latency(latency_comparison_map, csv_filename)

    if len(profile_metrics_map) > 0:
        logger.info("\n========================================")
        logger.info("========== TRT detail metrics ==========")
        logger.info("========================================")
        pp.pprint(profile_metrics_map)
        csv_filename = f"benchmark_ratio_{time_stamp}.csv"
        csv_filename = os.path.join(path, csv_filename)
        output_ratio(profile_metrics_map, csv_filename)


    logger.info("\n===========================================")
    logger.info("=========== System information  ===========")
    logger.info("===========================================")
    info = {}
    get_system_info(info)
    pp.pprint(info)
    csv_filename = os.path.join(path, f"system_info_{time_stamp}.csv")
    output_system_info(info, csv_filename)

    if fail_results:
        csv_filename = f"benchmark_fail_{time_stamp}.csv"
        csv_filename = os.path.join(path, csv_filename)
        output_fail(fail_results, csv_filename)

    if success_results:
        csv_filename = f"benchmark_success_{time_stamp}.csv"
        csv_filename = os.path.join(path, csv_filename)
        output_details(success_results, csv_filename)


if __name__ == "__main__":
    main()
