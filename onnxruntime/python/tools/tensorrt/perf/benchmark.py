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
from onnx import numpy_helper
from perf_utils import * 
import pprint
import torch

from BERTSquad import *
from Resnet50 import *
from Resnet101 import *
from Resnet152 import *
from FastRCNN import *
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
from Vgg import *
from Zfnet512 import *
from Retinanet import *
from Resnet101DucHdc import *
from ArcFace import *
# from SuperResolution import *
from FastNeural import *
from BiDAF import *
from GPT2 import *

logger = logging.getLogger('')

MODELS = {
    "bert-squad": (BERTSquad, "bert-squad"),
    "fast-rcnn": (FastRCNN, "fast-rcnn"),
    "mask-rcnn": (MaskRCNN, "mask-rcnn"),
    "ssd": (SSD, "ssd"),
    "tiny-yolov2": (TinyYolov2, "tiny-yolov2"),
    "tiny-yolov3": (TinyYolov3, "tiny-yolov3"),
    "resnet152": (Resnet152, "resnet152"),
    "inception-v2": (InceptionV2, "inception-v2"),
    "mobilenet-v2": (Mobilenet, "mobilenet-v2"),
    "zfnet512": (Zfnet512, "zfnet512"),
    "vgg19-bn": (Vgg, "vgg19-bn"),
    "resnet50": (Resnet50, "resnet50"),
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
    ##### "yolov3": (YoloV3, "yolov3"), # TRT runtime error
    "yolov4": (YoloV4, "yolov4"),
    "Resnet101-DUC": (Resnet101DucHdc, "Resnet101-DUC"),
    "Arc-Face": (ArcFace, "arc-face"),
    #### "Super-Resolution": (SuperResolution, "super-resolution"), # can't read output
    "Fast-Neural": (FastNeural, "Fast-Neural"),
    "BiDAF": (BiDAF, "BiDAF"),
    ### "GPT2": (GPT2, "GPT2"), # OOM
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


def inference_ort(model, ort_inputs, result_template, repeat_times, batch_size):

    runtimes = []
    for ort_input in ort_inputs:
        session = model.get_session() 

        inputs = model.get_ort_inputs(ort_input)
        outputs = model.get_ort_outputs()
        try:
            runtimes = runtimes + timeit.repeat(lambda: session.run(outputs, inputs), number=1, repeat=repeat_times)
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
    p = subprocess.Popen(["cat", "/usr/local/cuda/version.txt"], stdout=subprocess.PIPE) # (stdout, stderr)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    
    return stdout

def get_trt_version():
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
def load_onnx_model_zoo_test_data(path):
    print("Parsing inputs/outputs of test data in {} ...".format(path))
    p1 = subprocess.Popen(["find", path, "-name", "test_data_set*", "-type", "d"], stdout=subprocess.PIPE)
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
                input_data_pb.append(numpy_helper.to_array(tensor))
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
                output_data_pb.append(numpy_helper.to_array(tensor))
                print(np.array(output_data_pb[-1]).shape)
        outputs.append(output_data_pb)

        os.chdir(pwd)

    print('Loaded {} inputs successfully.'.format(len(inputs)))
    print('Loaded {} outputs successfully.'.format(len(outputs)))

    return inputs, outputs

def validate(all_ref_outputs, all_outputs, decimal):
    print('Reference {} results.'.format(len(all_ref_outputs)))
    print('Predicted {} results.'.format(len(all_outputs)))
    print('decimal {}'.format(decimal))
    # print(np.array(all_ref_outputs).shape)
    # print(np.array(all_outputs).shape)

    # if np.array(all_ref_outputs).shape != np.array(all_outputs).shape:
        # print("output result shares are not the same!")

    # print(all_ref_outputs)
    # print(all_outputs)
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
                    np.testing.assert_almost_equal(ref_o, o, decimal)
    except Exception as e:
        logger.error(e) 
        return False

    print('ONNX Runtime outputs are similar to reference outputs!')
    return True

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
        subprocess.Popen(["rm","-rf", f], stdout=subprocess.PIPE)

def remove_profiling_files(path):
    files = []
    p = subprocess.Popen(["find", path, "-name", "onnxruntime_profile*"], stdout=subprocess.PIPE)
    stdout, sterr = p.communicate()
    stdout = stdout.decode("ascii").strip()
    files = files + stdout.split("\n") 

    for f in files:
        subprocess.Popen(["rm","-rf", f], stdout=subprocess.PIPE)

def update_fail_model(model_name, ep, ep_model_fail_map):
    if not model_name in ep_model_fail_map:
        ep_model_fail_map[model_name] = [ep]
    else:
        if ep not in ep_model_fail_map[model_name]:
            ep_model_fail_map[model_name].append(ep)

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
    import onnxruntime
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout

    results = []
    latency_comparison_map = {} 
    ep_model_fail_map = {}
    profile_metrics_map = {}

    sys_info = {} 
    get_system_info(sys_info)

    ep_to_provider_list = {
        "CUDAExecutionProvider": ["CUDAExecutionProvider"],
        "TensorrtExecutionProvider": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        "TensorrtExecutionProvider_fp16": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    }

    if args.fp16:
        provider_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider","TensorrtExecutionProvider_fp16"]
    else:
        provider_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]

    # provider_list = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    # provider_list = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
    # provider_list = ["CUDAExecutionProvider"]
    # provider_list = ["TensorrtExecutionProvider"]

    # iterate models
    for name in models.keys():
        info = models[name] 
        model_class = info[0]
        path = info[1]
        latency_result = {}

        pwd = os.getcwd()
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        path = os.getcwd()
        
        # cleanup files before running a new inference
        remove_profiling_files(path)

        inputs = []
        ref_outputs = []
        ep_fail_set = set()
        ep_op_map = {} # map of ep -> operator
        profile_already_parsed = set()

        # iterate ep 
        for i in range(len(provider_list)):

            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"
            fp16 = False 
            ep = provider_list[i]
            model = None

            if "fp16" in ep:
                # No need to run TRT FP16 again if TRT already failed on previous run
                if "TensorrtExecutionProvider" in ep_fail_set:
                    continue

                os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
                fp16 = True 

            ep_ = ep_to_provider_list[ep][0]
            if (ep_ not in onnxruntime.get_available_providers()):
                logger.error("No {} support".format(ep_))
                continue
                
            # create onnxruntime inference session
            logger.info("\nInitializing {} with {}...".format(name, ep_to_provider_list[ep]))

            model = model_class(providers=ep_to_provider_list[ep])
            model_name = model.get_model_name()

            # read input/output of test data
            if not inputs or not ref_outputs:
                test_data_dir = model.get_onnx_zoo_test_data_dir()
                inputs, ref_outputs = load_onnx_model_zoo_test_data(test_data_dir)

            # these settings are temporary
            sequence_length = 1
            optimize_onnx = False
            batch_size = 1

            # enable profiling to generate profiling file for analysis
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True 
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            model.set_session_options(options)

            try: 
                model.create_session()
            except Exception as e:
                logger.error(e)
                ep_fail_set.add(ep)
                update_fail_model(model_name, ep, ep_model_fail_map)
                continue

            sess = model.get_session()
            sess.disable_fallback()


            logger.info("Start to inference {} with {} ...".format(model_name, ep))
            logger.info(sess.get_providers())
            logger.info("ORT_TENSORRT_FP16_ENABLE={}".format(os.environ["ORT_TENSORRT_FP16_ENABLE"]))

            if model.get_input_nodes():
                logger.info("Model inputs nodes:")
                for input_meta in model.get_input_nodes():
                    logger.info(input_meta)
            if model.get_output_nodes():
                logger.info("Model outputs nodes:")
                for output_meta in model.get_output_nodes():
                    logger.info(output_meta)

            # first, run inference and validate the result
            if ep in ['TensorrtExecutionProvider', 'CUDAExecutionProvider']:
                try:
                    model.set_inputs(inputs)
                    model.inference()
                    status = validate(ref_outputs, model.get_outputs(), model.get_decimal())
                    if not status:
                        ep_fail_set.add(ep)
                        update_fail_model(model_name, ep, ep_model_fail_map)
                        continue
                except Exception as e:
                    logger.error(e)
                    ep_fail_set.add(ep)
                    update_fail_model(model_name, ep, ep_model_fail_map)
                    continue

            # make sure profiling file will be generated by runing one inference
            inference_ort(model, inputs, {}, 2, batch_size)


            sess.end_profiling()
            logger.info(sess.get_providers())

            # for io_binding in [True, False]:
            for io_binding in [False]:
                result_template = {
                    "engine": "onnxruntime",
                    "version": onnxruntime.__version__,
                    "device": ep,
                    "optimizer": optimize_onnx,
                    "fp16": fp16,
                    "io_binding": io_binding,
                    "model_name": model_name,
                    "inputs": len(sess.get_inputs()),
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "datetime": str(datetime.now()),
                }

                result = None

                # measure inference
                if io_binding:
                    # can't do io-binding for BiDAF
                    if model.get_model_name() != 'BiDAF':
                        result = inference_ort_with_io_binding(model, inputs, result_template, args.test_times, batch_size)
                        latency_result[ep + "_io_binding"] = result["average_latency_ms"]
                else:
                    result = inference_ort(model, inputs, result_template, args.test_times, batch_size)
                    print(result)
                    latency_result[ep] = result["average_latency_ms"]


                if result:
                    logger.info(result)
                    results.append(result)

            latency_comparison_map[model_name] = copy.deepcopy(latency_result)
            
            metrics = get_profile_metrics(path, profile_already_parsed)
            if metrics:
                ep_op_map[ep] = metrics

            # end of ep


        # get TRT operators/latency percentage
        if len(ep_op_map) > 0:
            trt_op_map = None 
            trt_fp16_op_map = None 
            cuda_op_map = None 

            for ep, op_map in ep_op_map.items():  
                if ep == "CUDAExecutionProvider":
                    cuda_op_map = op_map
                elif ep == "TensorrtExecutionProvider":
                    trt_op_map = op_map
                elif ep == "TensorrtExecutionProvider_fp16":
                    trt_fp16_op_map = op_map

            if trt_op_map:
                ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_op_map)
                profile_metrics_map[model_name] = {}
                profile_metrics_map[model_name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt 
                if cuda_op_map:
                    total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_op_map, cuda_op_map)
                    profile_metrics_map[model_name]['total_ops'] = total_ops
                    profile_metrics_map[model_name]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt 
                    profile_metrics_map[model_name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt 

            if trt_fp16_op_map:
                ratio_of_execution_time_in_trt = calculate_trt_latency_percentage(trt_fp16_op_map)
                model_name_ = model_name + "(TRT FP16)"
                profile_metrics_map[model_name_] = {}
                profile_metrics_map[model_name_]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt 
                if cuda_op_map:
                    total_ops_in_trt, total_ops, ratio_of_ops_in_trt = calculate_trt_op_percentage(trt_fp16_op_map, cuda_op_map)
                    profile_metrics_map[model_name_]['total_ops'] = total_ops
                    profile_metrics_map[model_name_]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt 
                    profile_metrics_map[model_name_]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt 


        # analyze profiling files 
        # if len(ep_fail_set) == 0:
            # presults = analyze_profiling_file(path)
            # for i in range(len(presults)):
                # result = presults[i]
                # total_ops_in_trt = result[0]
                # total_ops = result[1]
                # ratio_of_ops_in_trt = result[2]
                # ratio_of_execution_time_in_trt =  result[3]
                
                # name = model_name + " (TRT fp16)" if i == 0 else model_name
                # profile_metrics_map[name] = {}
                # profile_metrics_map[name]['total_ops_in_trt'] = total_ops_in_trt
                # profile_metrics_map[name]['total_ops'] = total_ops
                # profile_metrics_map[name]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt 
                # profile_metrics_map[name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt 

        cleanup_files()
        os.chdir(pwd)

        # end of model

    return results, latency_comparison_map, ep_model_fail_map, profile_metrics_map

def add_improvement_information(latency_comparison_map):
    for key, value in latency_comparison_map.items():
        if not ('TensorrtExecutionProvider' in value and 'CUDAExecutionProvider' in value):
            continue

        trt_latency = float(value['TensorrtExecutionProvider'])
        cuda_latency = float(value['CUDAExecutionProvider'])
        gain = (cuda_latency - trt_latency)*100/trt_latency
        value["Tensorrt_gain(%)"] = "{:.2f} %".format(gain) 

        if "TensorrtExecutionProvider_fp16" in value:
            trt_fp16_latency = float(value['TensorrtExecutionProvider_fp16'])
            gain = (cuda_latency - trt_fp16_latency)*100/trt_fp16_latency
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

def output_latency(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["Model",
                        "CUDA latency (ms)",
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
                        "% TRT execution time"]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        for key, value in results.items():
            row = [key,
                   value['total_ops_in_trt'] if 'total_ops_in_trt' in value else "  ",
                   value['total_ops'] if 'total_ops' in value else "  ",
                   value['ratio_of_ops_in_trt'] if 'ratio_of_ops_in_trt' in value else "  ",
                   value['ratio_of_execution_time_in_trt'] if 'ratio_of_execution_time_in_trt' in value else "  ",
                   ]
            csv_writer.writerow(row)
            

    logger.info(f"Tensorrt ratio metrics are saved to csv file: {csv_filename}")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--detail_csv", required=False, default=None, help="CSV file for saving detail results.")
    parser.add_argument("-r", "--ratio_csv", required=False, default=None, help="CSV file for saving detail results.")
    parser.add_argument("-l", "--latency_csv", required=False, default=None, help="CSV file for saving detail results.")

    parser.add_argument("--fp16", required=False, default=True, action="store_true", help="Use FP16 to accelerate inference")

    parser.add_argument("--fp32", required=False, action="store_true", help="Use FP32 to accelerate inference")

    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=8,
                        type=int,
                        help="Number of repeat times to get average inference latency.")

    # parser.add_argument("-r", "--result_csv", required=False, default=None, help="CSV file for saving summary results.")

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
    results, latency_comparison_map, failing_models, profile_metrics_map = run_onnxruntime(args)
    perf_end_time = datetime.now()

    logger.info("\nTotal time for running/profiling all models: {}".format(perf_end_time - perf_start_time))
    logger.info(list(MODELS.keys()))

    logger.info("\nTotal models: {}".format(len(MODELS)))
    logger.info("Fail models: {}".format(len(failing_models)))
    logger.info("Models FAIL/SUCCESS: {}/{}".format(len(failing_models), len(MODELS) - len(failing_models)))

    if len(failing_models) > 0:
        logger.info("\n============================================")
        logger.info("========== Failing Models/EPs ==============")
        logger.info("============================================")
        logger.info(failing_models)

    if latency_comparison_map:
        logger.info("\n=========================================")
        logger.info("=========== CUDA/TRT latency  ===========")
        logger.info("=========================================")
        add_improvement_information(latency_comparison_map)
        pp.pprint(latency_comparison_map)
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = args.latency_csv or f"benchmark_latency_{time_stamp}.csv"
        output_latency(latency_comparison_map, csv_filename)

    if len(profile_metrics_map) > 0:
        logger.info("\n========================================")
        logger.info("========== TRT detail metrics ==========")
        logger.info("========================================")
        pp.pprint(profile_metrics_map)
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = args.ratio_csv or f"benchmark_ratio_{time_stamp}.csv"
        output_ratio(profile_metrics_map, csv_filename)


    logger.info("\n===========================================")
    logger.info("=========== System information  ===========")
    logger.info("===========================================")
    info = {}
    get_system_info(info)
    pp.pprint(info)

    if results:
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = args.detail_csv or f"benchmark_detail_{time_stamp}.csv"
        output_details(results, csv_filename)


if __name__ == "__main__":
    main()
