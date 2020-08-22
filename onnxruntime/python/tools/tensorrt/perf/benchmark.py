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

sys.path.append('./models/')
from BERTSquad import *
from Resnet18v1 import *
from Resnet18v2 import *
from Resnet50v1 import *
from Resnet50v2 import *
from Resnet101 import *
from Resnet152v1 import *
from Resnet152v2 import *
from FasterRCNN import *
from MaskRCNN import *
from SSD import *
from InceptionV1 import *
from InceptionV2 import *
from Mobilenet import *
from ShufflenetV2 import *
from ShufflenetV1 import *
from Squeezenet import *
from EmotionFerplus import *
from Googlenet import *
from Alexnet import *
from Caffenet import *
from Densenet import *
from RcnnIlsvrc13 import *
from TinyYolov2 import *
from TinyYolov3 import *
from YoloV3 import *
from YoloV4 import *
from Vgg19bn import *
from Vgg16 import *
from Zfnet512 import *
from Retinanet import *
from Resnet101DucHdc import *
from ArcFace import *
# from SuperResolution import *
from FastNeural import *
from BiDAF import *
from GPT2 import *
from VehicleDetector import *
from TwoStageProductReco import *
from SeresnextGeneral import *
from YoloV2V2LOGO import *
from YoloV2V3 import *
from YoloV3PyTorch import *
# import torch

debug = False
sys.path.append('.')

logger = logging.getLogger('')

########################################
#
# model name: (model class, model path)
#
########################################
MODELS = {
    "bert-squad": (BERTSquad, "bert-squad"),
    "faster-rcnn": (FasterRCNN, "faster-rcnn"),
    "mask-rcnn": (MaskRCNN, "mask-rcnn"),
    "ssd": (SSD, "ssd"),
    "tiny-yolov2": (TinyYolov2, "tiny-yolov2"),
    "tiny-yolov3": (TinyYolov3, "tiny-yolov3"),
    "resnet152v1": (Resnet152v1, "resnet152v1"),
    "resnet152v2": (Resnet152v2, "resnet152v2"),
    "inception-v2": (InceptionV2, "inception-v2"),
    "mobilenet-v2": (Mobilenet, "mobilenet-v2"),
    "zfnet512": (Zfnet512, "zfnet512"),
    "vgg16": (Vgg16, "vgg16"),
    "vgg19-bn": (Vgg19bn, "vgg19-bn"),
    "resnet18v1": (Resnet18v1, "resnet18v1"),
    "resnet18v2": (Resnet18v2, "resnet18v2"),
    "resnet50v1": (Resnet50v1, "resnet50v1"),
    "resnet50v2": (Resnet50v2, "resnet50v2"),
    "resnet101": (Resnet101, "resnet101"),
    "inception-v1": (InceptionV1, "inception-v1"),
    "shufflenet-v1": (ShufflenetV1, "shufflenet-v1"),
    "shufflenet-v2": (ShufflenetV2, "shufflenet-v2"),
    "squeezenet1.1": (Squeezenet, "squeezenet1.1"),
    "emotion-ferplus": (EmotionFerplus, "emotion-ferplus"),
    "bvlc-googlenet": (Googlenet, "bvlc-googlenet"),
    "bvlc-alexnet": (Alexnet, "bvlc-alexnet"),
    "bvlc-caffenet": (Caffenet, "bvlc-caffenet"),
    "bvlc-rcnn-ilvscr13": (RcnnIlsvrc13, "bvlc-rcnn-ilvscr13"),
    "retinanet": (Retinanet, "retinanet"),
    "densenet": (Densenet, "densenet"),
    "yolov3": (YoloV3, "yolov3"),
    "yolov4": (YoloV4, "yolov4"),
    "Resnet101-DUC": (Resnet101DucHdc, "Resnet101-DUC"),
    "Arc-Face": (ArcFace, "arc-face"),
    # #### "Super-Resolution": (SuperResolution, "super-resolution"), # can't read output
    "Fast-Neural": (FastNeural, "Fast-Neural"),
    "BiDAF": (BiDAF, "BiDAF"),
    "GPT2": (GPT2, "GPT2"),
}

# Additional models that onwed by Custom Vision Service
CVS_MODELS = {
    "vehicle-detector": (VehicleDetector, "vehicle-detector"),
    "two-stage-product-reco": (TwoStageProductReco, "two-stage-product-reco"),
    "seresnext-general": (SeresnextGeneral, "seresnext-general"),
    "yolov2v2-logo": (YoloV2V2LOGO, "yolov2v2-logo"),
    "yolov2v3": (YoloV2V3, "yolov2v3"),
    "yolov3-pytorch": (YoloV3PyTorch, "yolov3-pytorch"),
}

ep_to_provider_list = {
    "CPUExecutionProvider": ["CPUExecutionProvider"],
    "CUDAExecutionProvider": ["CUDAExecutionProvider"],
    "CUDAExecutionProvider_fp16": ["CUDAExecutionProvider"],
    "TensorrtExecutionProvider": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    "TensorrtExecutionProvider_fp16": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
}


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


def inference_ort(args, model, ep, ort_inputs, result_template, repeat_times, batch_size):

    runtimes = []
    for ort_input in ort_inputs:
        session = model.get_session()

        inputs = model.get_ort_inputs(ort_input)
        outputs = model.get_ort_outputs()
        try:
            if args.input_data == "random":
                repeat_times = 1

            if ep in ["TensorrtExecutionProvider", "TensorrtExecutionProvider_fp16"]:
                runtime = timeit.repeat(lambda: session.run(outputs, inputs), number=1, repeat=repeat_times+1)
                runtime[:] = runtime[1:]
            else:
                runtime = timeit.repeat(lambda: session.run(outputs, inputs), number=1, repeat=repeat_times)

            runtimes += runtime

        except Exception as e:
            logger.error(e)
            return None

    print(runtimes)

    result = {}
    result.update(result_template)
    result.update({"io_binding": False})
    result.update(get_latency_result(runtimes, batch_size))
    return result

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

def get_cuda_version_old():
    p = subprocess.Popen(["cat", "/usr/local/cuda/version.txt"], stdout=subprocess.PIPE) # (stdout, stderr)
    stdout, sterr = p.communicate()
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

def get_trt_version_old():
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
def load_onnx_model_zoo_test_data(path, data_type="fp32"):
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

                print(np.array(input_data_pb[-1]).shape)
        inputs.append(input_data_pb)

        # load outputs
        p1 = subprocess.Popen(["find", ".", "-name", "output*"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
        stdout, sterr = p2.communicate()
        stdout = stdout.decode("ascii").strip()
        output_data = stdout.split("\n")
        print(output_data)

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

        os.chdir(pwd)

    print('Loaded {} inputs successfully.'.format(len(inputs)))
    print('Loaded {} outputs successfully.'.format(len(outputs)))

    return inputs, outputs

def generate_onnx_model_random_input(test_times, ref_input):

    test_input_data = []

    for i in range(test_times):

        input_data = []
        for tensor in ref_input:
            shape = tensor.shape
            dtype = tensor.dtype
            new_tensor = np.random.random_sample(shape).astype(dtype)
            input_data.append(new_tensor)
        test_input_data.append(input_data)

    return test_input_data

def validate(all_ref_outputs, all_outputs, decimal):
    print('Reference {} results.'.format(len(all_ref_outputs)))
    print('Predicted {} results.'.format(len(all_outputs)))
    print('decimal {}'.format(decimal))
    # print(np.array(all_ref_outputs).shape)
    # print(np.array(all_outputs).shape)

    # if np.array(all_ref_outputs).shape != np.array(all_outputs).shape:
        # print("output result shares are not the same!")

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

    # result["running_mode"] = args.running_mode
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

def run_onnxruntime(args, models=MODELS):

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
        provider_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CUDAExecutionProvider_fp16", "TensorrtExecutionProvider_fp16"]
    else:
        provider_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]


    if args.model_zoo == "cvs":
        models.update(CVS_MODELS)

    #######################
    # iterate model
    #######################
    for name in models.keys():
        info = models[name]
        model_class = info[0]
        path = os.path.join(os.getcwd(), 'models', info[1])
        latency_result = {}

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
        ep_to_ep_op_map = {} # ep -> { ep -> operator }
        profile_already_parsed = set()


        #######################
        # iterate ep
        #######################
        for ep in provider_list:

            if skip_ep(name, ep, model_ep_fail_map):
                continue

            ep_ = ep_to_provider_list[ep][0]
            if (ep_ not in onnxruntime.get_available_providers()):
                logger.error("No {} support".format(ep_))
                continue

            if "fp16" in ep:
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
                fp16 = True
                logger.info("\nInitializing {} with float16 enabled to run on {} ...".format(name, ep))
            else:
                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"
                fp16 = False
                logger.info("\nInitializing {} to run on {} ...".format(name, ep))

            # create model instance
            model = model_class(providers=ep_to_provider_list[ep])
            if ep == "CUDAExecutionProvider_fp16":
                model.convert_model_from_float_to_float16()
            model_name = model.get_model_name()
            model.set_model_zoo_source(args.model_zoo)

            # read input/output of test data
            if fp16 and ep == "CUDAExecutionProvider_fp16":
                if not inputs_fp16 or not ref_outputs_fp16:
                    if args.model_zoo ==  "onnx":
                        test_data_dir = model.get_onnx_zoo_test_data_dir()
                    elif args.model_zoo == "cvs":
                        test_data_dir = model.get_cvs_model_test_data_dir()
                    inputs_fp16, ref_outputs_fp16 = load_onnx_model_zoo_test_data(test_data_dir, "fp16")

                inputs = inputs_fp16
                ref_outputs = ref_outputs_fp16
            else:
                if not inputs_fp32 or not ref_outputs_fp32:
                    if args.model_zoo ==  "onnx":
                        test_data_dir = model.get_onnx_zoo_test_data_dir()
                    elif args.model_zoo == "cvs":
                        test_data_dir = model.get_cvs_model_test_data_dir()
                    inputs_fp32, ref_outputs_fp32 = load_onnx_model_zoo_test_data(test_data_dir)

                inputs = inputs_fp32
                ref_outputs = ref_outputs_fp32


            if args.input_data == "random":
                inputs = generate_onnx_model_random_input(args.test_times, inputs[0])

            #######################################
            # benchmark or validation
            #######################################
            if args.running_mode == 'benchmark':

                try:
                    # create onnxruntime inference session
                    if args.model_zoo ==  "onnx":
                        model.create_session()
                    elif args.model_zoo == "cvs":
                        model.create_session(model.get_cvs_model_path())

                except Exception as e:
                    logger.error(e)
                    # update_fail_model(model_ep_fail_map, fail_results, args, name, ep, e)
                    continue

                sess = model.get_session()
                logger.info("Start to inference {} with {} ...".format(model_name, ep))
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
                    "optimizer": False,
                    "fp16": fp16,
                    "io_binding": False,
                    "model_name": model_name,
                    "inputs": len(sess.get_inputs()),
                    "batch_size": batch_size,
                    "sequence_length": 1,
                    "datetime": str(datetime.now()),}

                result = inference_ort(args, model, ep, inputs, result_template, args.test_times, batch_size)
                success_results.append(result)
                logger.info(result)

                latency_result[ep] = result["average_latency_ms"]
                latency_comparison_map[model_name] = copy.deepcopy(latency_result)


            elif args.running_mode == 'validate':

                # enable profiling to generate profiling file for analysis
                options = onnxruntime.SessionOptions()
                options.enable_profiling = True
                options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                model.set_session_options(options)

                try:
                    # create onnxruntime inference session
                    if args.model_zoo ==  "onnx":
                        model.create_session()
                    elif args.model_zoo == "cvs":
                        model.create_session(model.get_cvs_model_path())

                except Exception as e:
                    logger.error(e)
                    update_fail_model(model_ep_fail_map, fail_results, args, name, ep, 'runtime error', e)
                    continue


                sess = model.get_session()
                sess.disable_fallback()

                logger.info("Start to inference {} with {} ...".format(model_name, ep))
                logger.info(sess.get_providers())

                if sess:
                    logger.info("Model inputs nodes:")
                    for input_meta in sess.get_inputs():
                        logger.info(input_meta)
                    logger.info("Model outputs nodes:")
                    for output_meta in sess.get_outputs():
                        logger.info(output_meta)

                # run inference and validate the result
                try:
                    model.set_inputs(inputs)
                    model.inference()

                    # lower accuracy exepectation if using FP16
                    if ep == 'CUDAExecutionProvider_fp16' or ep == 'TensorrtExecutionProvider_fp16':
                        decimal = 0
                    else:
                        decimal = model.get_decimal()

                    status = validate(ref_outputs, model.get_outputs(), decimal)
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
                model.inference()

                sess.end_profiling()
                time.sleep(1) # avoid to generate same profile file name

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

            if trt_op_map:
                total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_op_map)
                profile_metrics_map[model_name] = {}
                profile_metrics_map[model_name]['total_trt_execution_time'] = total_trt_execution_time
                profile_metrics_map[model_name]['total_execution_time'] = total_execution_time
                profile_metrics_map[model_name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt
                if cuda_op_map:
                    total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_op_map, cuda_op_map)
                    profile_metrics_map[model_name]['total_ops_in_trt'] = total_ops_in_trt
                    profile_metrics_map[model_name]['total_ops'] = total_ops
                    profile_metrics_map[model_name]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt

            if trt_fp16_op_map:
                total_trt_execution_time, total_execution_time, ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_fp16_op_map)
                model_name_ = model_name + " (FP16)"
                profile_metrics_map[model_name_] = {}
                profile_metrics_map[model_name_]['total_trt_execution_time'] = total_trt_execution_time
                profile_metrics_map[model_name_]['total_execution_time'] = total_execution_time
                profile_metrics_map[model_name_]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt
                if cuda_fp16_op_map:
                    total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_fp16_op_map, cuda_op_map)
                    profile_metrics_map[model_name_]['total_ops_in_trt'] = total_ops_in_trt
                    profile_metrics_map[model_name_]['total_ops'] = total_ops
                    profile_metrics_map[model_name_]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt

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



        cleanup_files()
        os.chdir(pwd)

        # end of model

    return success_results, fail_results, latency_comparison_map, model_ep_fail_map, profile_metrics_map

def add_improvement_information(latency_comparison_map):
    for key, value in latency_comparison_map.items():
        if not ('TensorrtExecutionProvider' in value and 'CUDAExecutionProvider' in value):
            continue

        trt_latency = float(value['TensorrtExecutionProvider'])
        cuda_latency = float(value['CUDAExecutionProvider'])
        gain = (cuda_latency - trt_latency)*100/cuda_latency
        value["Tensorrt_gain(%)"] = "{:.2f} %".format(gain)

        if "TensorrtExecutionProvider_fp16" in value and "CUDAExecutionProvider_fp16" in value:
            trt_fp16_latency = float(value['TensorrtExecutionProvider_fp16'])
            cuda_fp16_latency = float(value['CUDAExecutionProvider_fp16'])
            gain = (cuda_fp16_latency - trt_fp16_latency)*100/cuda_fp16_latency
            value["Tensorrt_fp16_gain(%)"] = "{:.2f} %".format(gain)

def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "fp16", "optimizer", "io_binding", "model_name", "inputs", "batch_size",
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
                        "CUDA latency (ms)",
                        "CUDA fp16 latency (ms)",
                        "CUDA latency with io_binding (ms)",
                        "TRT latency (ms)",
                        "TRT latency with io_binding (ms)",
                        "TRT latency with fp16 (ms)",
                        "TRT latency with fp16 and io_binding (ms)",
                        "TRT gain (%)",
                        "TRT f16 gain(%)"]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        for key, value in results.items():
            row = [key,
                   value['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in value else "  ",
                   value['CUDAExecutionProvider_fp16'] if 'CUDAExecutionProvider_fp16' in value else "  ",
                   value['CUDAExecutionProvider_io_binding'] if 'CUDAExecutionProvider_io_binding' in value else "  ",
                   value['TensorrtExecutionProvider'] if 'TensorrtExecutionProvider' in value else "  ",
                   value['TensorrtExecutionProvider_io_binding'] if 'TensorrtExecutionProvider_io_binding' in value else "  ",
                   value['TensorrtExecutionProvider_fp16'] if 'TensorrtExecutionProvider_fp16' in value else "  ",
                   value['TensorrtExecutionProvider_fp16_io_binding'] if 'TensorrtExecutionProvider_fp16_io_binding' in value else "  ",
                   value['Tensorrt_gain(%)'] if 'Tensorrt_gain(%)' in value else "  ",
                   value['Tensorrt_fp16_gain(%)'] if 'Tensorrt_fp16_gain(%)' in value else "  "
                   ]
            csv_writer.writerow(row)

    logger.info(f"CUDA/TRT latency comparison are saved to csv file: {csv_filename}")

def output_ratio(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["Model",
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

    parser.add_argument("-m", "--model_zoo", required=False, default="onnx", choices=["onnx", "cvs"], help="Source of the models.")

    parser.add_argument("-r", "--running_mode", required=False, default="benchmark", choices=["validate", "benchmark"], help="Testing mode.")

    parser.add_argument("-i", "--input_data", required=False, default="zoo", choices=["zoo", "random"], help="source of input data.")

    parser.add_argument("--fp16", required=False, default=True, action="store_true", help="Inlcude Float16 into benchmarking.")

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

    perf_start_time = datetime.now()
    success_results, fail_results, latency_comparison_map, failing_models, profile_metrics_map = run_onnxruntime(args)
    perf_end_time = datetime.now()

    logger.info("\nTotal time for running/profiling all models: {}".format(perf_end_time - perf_start_time))
    logger.info(list(MODELS.keys()))

    logger.info("\nTotal models: {}".format(len(MODELS)))
    logger.info("Fail models: {}".format(len(failing_models)))
    logger.info("Models FAIL/SUCCESS: {}/{}".format(len(failing_models), len(MODELS) - len(failing_models)))

    path = "perf_result"
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
