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
from onnx import numpy_helper
from perf_utils import analyze_profiling_file
import pprint

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
from RcnnIlsvrc13 import *
from TinyYolov2 import *
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
    "resnet50": (Resnet50, "resnet50"),
    "resnet101": (Resnet101, "resnet101"),
    "resnet152": (Resnet152, "resnet152"),
    "fast-rcnn": (FastRCNN, "fast-rcnn"),
    "mask-rcnn": (MaskRCNN, "mask-rcnn"),
    "ssd": (SSD, "ssd"),
    "inception-v1": (InceptionV1, "inception-v1"),
    "inception-v2": (InceptionV2, "inception-v2"),
    "mobilenet-v2": (Mobilenet, "mobilenet-v2"),
    "shufflenet-v1": (ShufflenetV1, "shufflenet-v1"),
    "shufflenet-v2": (ShufflenetV2, "shufflenet-v2"),
    "squeezenet1.1": (Squeezenet, "squeezenet1.1"),
    "emotion-ferplus": (EmotionFerplus, "emotion-ferplus"),
    "bvlc-googlenet": (Googlenet, "bvlc-googlenet"),
    "bvlc-alexnet": (Alexnet, "bvlc-alexnet"),
    "bvlc-caffenet": (Alexnet, "bvlc-caffenet"),
    "bvlc-rcnn-ilvscr13": (RcnnIlsvrc13, "bvlc-rcnn-ilvscr13"),
    "tiny-yolov2": (TinyYolov2, "tiny-yolov2"),
    "vgg19-bn": (Vgg, "vgg19-bn"),
    "zfnet512": (Zfnet512, "zfnet512"),
    "retinanet": (Retinanet, "retinanet"),
    "yolov3": (YoloV3, "yolov3"),
    "yolov4": (YoloV4, "yolov4"),
    "Resnet101-DUC": (Resnet101DucHdc, "Resnet101-DUC"),
    "Arc-Face": (ArcFace, "arc-face"),
    # "Super-Resolution": (SuperResolution, "super-resolution"), # can't read output
    "Fast-Neural": (FastNeural, "Fast-Neural"),
    "BiDAF": (BiDAF, "BiDAF"),
    # "GPT2": (GPT2, "GPT2"),
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


def inference_ort(model, inputs, result_template, repeat_times, batch_size):
    result = {}

    try:
        runtimes = timeit.repeat(lambda: model.inference(inputs), number=1, repeat=repeat_times+1)
    except Exception as e:
        logger.error(e)
        return None

    runtimes[:] = runtimes[1:] # we intentionally skip the first run due to TRT is expensive on first run
    result.update(result_template)
    result.update({"io_binding": False})
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
        import re
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
# inputs: [[[test_data_0_input_0.pb], [test_data_0_input_1.pb] ...], [[test_data_1_input_0.pb], [test_data_1_input_1.pb] ...] ...]
# outputs: [[[test_data_0_output_0.pb], [test_data_0_output_1.pb] ...], [[test_data_1_output_0.pb], [test_data_1_output_1.pb] ...] ...]
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
        info.error(e) 
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
        ep_model_fail_map[model_name].append(ep)

def get_system_info(info):
    import re
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

    results = []
    latency_comparison_map = {} 
    ep_model_fail_map = {}
    profile_metrics_map = {}

    sys_info = {} 
    get_system_info(sys_info)

    provider_list = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    # provider_list = ["CPUExecutionProvider"]

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

        for i in range(len(provider_list)):
            ep = provider_list[i]
            ep_fail_flag = False

            if (ep not in onnxruntime.get_available_providers()):
                logger.error("No {} support".format(ep))
                continue
                
            start_time = datetime.now()

            # create onnxruntime inference session
            logger.info("\nInitializing {} with {}...".format(name, provider_list[i:]))
            model = model_class(providers=provider_list[i:])
            model_name = model.get_model_name()

            # read input/output of test data
            if not inputs or not ref_outputs:
                test_data_dir = model.get_onnx_zoo_test_data_dir()
                inputs, ref_outputs = load_onnx_model_zoo_test_data(test_data_dir)


            # these settings are temporary
            sequence_length = 1
            optimize_onnx = False
            batch_size = 1

            # enable profiling
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True 
            model.set_session_options(options)
            try: 
                model.create_session()
            except Exception as e:
                logger.error(e)
                ep_fail_flag = True
                update_fail_model(model_name, ep, ep_model_fail_map)
                continue

            sess = model.get_session()

            result_template = {
                "engine": "onnxruntime",
                "version": onnxruntime.__version__,
                "device": ep,
                "optimizer": optimize_onnx,
                "fp16": args.fp16,
                "io_binding": False,
                "model_name": model_name,
                "inputs": len(sess.get_inputs()),
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "datetime": str(datetime.now()),
                "total_time": None,
            }

            logger.info("Start to inference {} with {} ...".format(model_name, ep))
            logger.info(sess.get_providers())


            if model.get_input_name():
                logger.info("Model inputs name:")
                for input_meta in model.get_input_name():
                    logger.info(input_meta)

            # model inference
            result = inference_ort(model, inputs, result_template, args.test_times, batch_size)
            if not result: 
                ep_fail_flag = True
                update_fail_model(model_name, ep, ep_model_fail_map)
                continue

            end_time = datetime.now()
            result["total_time"] = str(end_time - start_time) 

            status = validate(ref_outputs, model.get_outputs(), model.get_decimal())
            if not status:
                ep_fail_flag = True
                update_fail_model(model_name, ep, ep_model_fail_map)
                continue

            latency_result[ep] = result["average_latency_ms"]

            logger.info(result)
            results.append(result)
            latency_comparison_map[model_name] = copy.deepcopy(latency_result)


        sess = model.get_session()
        sess.end_profiling()

        if not ep_fail_flag:
            total_ops_in_trt, total_ops, ratio_of_ops_in_trt, ratio_of_execution_time_in_trt = analyze_profiling_file(path)
            profile_metrics_map[model_name] = {}
            profile_metrics_map[model_name]['total_ops_in_trt'] = total_ops_in_trt
            profile_metrics_map[model_name]['total_ops'] = total_ops
            profile_metrics_map[model_name]['ratio_of_ops_in_trt'] = ratio_of_ops_in_trt 
            profile_metrics_map[model_name]['ratio_of_execution_time_in_trt'] = ratio_of_execution_time_in_trt 

        cleanup_files()
        os.chdir(pwd)

    return results, latency_comparison_map, ep_model_fail_map, profile_metrics_map

def output_details(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = [
            "engine", "version", "device", "fp16", "optimizer", "io_binding", "model_name", "inputs", "batch_size",
            "sequence_length", "datetime", "total_time", "test_times", "QPS", "average_latency_ms", "latency_variance",
            "latency_90_percentile", "latency_95_percentile", "latency_99_percentile"
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in results:
            csv_writer.writerow(result)

    logger.info(f"Detail results are saved to csv file: {csv_filename}")

def output_latency_comparison(results, csv_filename):
    with open(csv_filename, mode="a", newline='') as csv_file:
        column_names = ["Model", "CUDA latency (ms)", "TRT latency (ms)"]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        for key, value in results.items():
            row = [key,
                   value['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in value else "  ",
                   value['TensorrtExecutionProvider'] if 'TensorrtExecutionProvider' in value else "  "]
            csv_writer.writerow(row)
            

    logger.info(f"Latency comparison are saved to csv file: {csv_filename}")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--detail_csv", required=False, default=None, help="CSV file for saving detail results.")

    parser.add_argument("--fp16", required=False, action="store_true", help="Use FP16 to accelerate inference")

    parser.add_argument("--fp32", required=False, action="store_true", help="Use FP32 to accelerate inference")

    parser.add_argument("-t",
                        "--test_times",
                        required=False,
                        default=10,
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
        logger.info("\nFailing EPs and Models:")
        logger.info(failing_models)

    if len(profile_metrics_map) > 0:
        logger.info("\nTRT related metrics:")
        pp.pprint(profile_metrics_map)

    if latency_comparison_map:
        logger.info("\nCUDA/TRT inference time comparison:")
        pp.pprint(latency_comparison_map)


    if results:
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = args.detail_csv or f"benchmark_detail_{time_stamp}.csv"
        output_details(results, csv_filename)

    if latency_comparison_map:
        csv_filename = args.detail_csv or f"benchmark_latency_{time_stamp}.csv"
        output_latency_comparison(latency_comparison_map, csv_filename)

    info = {}
    get_system_info(info)
    pp.pprint(info)
    


if __name__ == "__main__":
    main()
