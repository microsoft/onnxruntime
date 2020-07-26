import subprocess
import json
import pprint
import logging
import coloredlogs
# import os

debug = True 
debug_verbose = False 

def parse_single_file(f):
    import re
    data = json.load(f)

    model_run_flag = False 
    first_run_flag = True
    provider_op_map = {}  # ep -> map of operators and duration
    provider_op_map_first_run = {} # ep -> map of operators and duration

    for row in data:
        if not "cat" in row:
            continue

        if row["cat"] == "Session":
            if "name" in row and row["name"] == "model_run":
                if not first_run_flag:
                    break

                model_run_flag = True
                first_run_flag = False 

        elif row["cat"] == "Node":
            if "name" in row and "args" in row and re.search(".*kernel_time", row["name"]):
                args = row["args"]
                
                if not "op_name" in args or not "provider" in args:
                    continue
                
                provider = args["provider"]

                if first_run_flag:
                    if provider not in provider_op_map_first_run:
                        provider_op_map_first_run[provider] = {}

                    op_map = provider_op_map_first_run[provider]
                    
                    if row["name"] in op_map:
                        provider_op_map[provider] = {}
                        op_map = provider_op_map[provider]
                        op_map[row["name"]] = row["dur"]
                        provider_op_map[provider] = op_map
                    else:
                        op_map[row["name"]] = row["dur"]
                        provider_op_map_first_run[provider] = op_map
                else:
                    if provider not in provider_op_map:
                        provider_op_map[provider] = {}

                    op_map = provider_op_map[provider]
                    
                    # avoid duplicated metrics 
                    if not row["name"] in op_map:
                        op_map[row["name"]] = row["dur"]
                        provider_op_map[provider] = op_map


    if debug_verbose:
        pp = pprint.PrettyPrinter(indent=4)
        print("------First run ops map (START)------")
        pp.pprint(provider_op_map_first_run)
        print("------First run ops map (END) ------")
        print("------Second run ops map (START)------")
        pp.pprint(provider_op_map)
        print("------Second run ops map (END) ------")

    if model_run_flag:
        return provider_op_map 
    
    return None

#
# Return: total ops executed in TRT,
#         total ops,
#         ratio of ops executed in TRT,
#         ratio of execution time in TRT
#
def calculate_metrics(trt_op_map, cuda_op_map):

    # % of TRT ops
    total_ops = 0
    total_cuda_and_cpu_ops = 0 
    for ep in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        if ep in cuda_op_map:
            op_map = cuda_op_map[ep]
            total_ops += len(op_map)

        if ep in trt_op_map:
            op_map = trt_op_map[ep]
            total_cuda_and_cpu_ops += len(op_map)

    if total_ops == 0:
        print("Error ...")
        raise

    if len(trt_op_map) == 0:
        total_cuda_and_cpu_ops = total_ops

    #
    # equation of % TRT ops: 
    # (total ops in cuda json - cuda and cpu ops in trt json)/ total ops in cuda json
    #
    ratio_of_ops_in_trt = (total_ops - total_cuda_and_cpu_ops) / total_ops
    if debug:
        print("total_cuda_and_cpu_ops: {}".format(total_cuda_and_cpu_ops))
        print("total_ops: {}".format(total_ops))
        print("ratio_of_ops_in_trt: {}".format(ratio_of_ops_in_trt))


    # % of TRT execution time
    total_execution_time = 0
    total_trt_execution_time = 0 
    for ep in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
        if ep in trt_op_map:
            op_map = trt_op_map[ep]

            total_time = 0
            for key, value in op_map.items():
                total_time += int(value)

            if ep == "TensorrtExecutionProvider":
                total_trt_execution_time = total_time

            total_execution_time += total_time



    if total_execution_time == 0:
        ratio_of_trt_execution_time = 0
    else:
        ratio_of_trt_execution_time = total_trt_execution_time / total_execution_time

    if debug:
        print("total_trt_execution_time: {}".format(total_trt_execution_time))
        print("total_execution_time: {}".format(total_execution_time))
        print("ratio_of_trt_execution_time: {}".format(ratio_of_trt_execution_time))

    return ((total_ops - total_cuda_and_cpu_ops), total_ops, ratio_of_ops_in_trt, ratio_of_trt_execution_time)


def analyze_profiling_file(path):
    print("Analying profiling files in {} ...".format(path))
    p1 = subprocess.Popen(["find", path, "-name", "onnxruntime_profile*"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    profiling_file_dir = stdout.split("\n") 
    print(profiling_file_dir)

    pp = pprint.PrettyPrinter(indent=4)

    data = []
    for profiling_file in profiling_file_dir:
        with open(profiling_file) as f:
            op_map = parse_single_file(f)
            if op_map:
                data.append(op_map)

    trt_op_map = {}
    trt_fp16_op_map = {}
    cuda_op_map = {}
    cpu_op_map = {}

    list_of_trt_op_map = []
    list_of_cuda_op_map = []
    list_of_cpu_op_map = []

    for op_map in data:
        if "TensorrtExecutionProvider" in op_map:
            list_of_trt_op_map.append(op_map)
        elif  "CUDAExecutionProvider" in op_map:
            list_of_cuda_op_map.append(op_map)
        elif "CPUExecutionProvider" in op_map:
            list_of_cpu_op_map.append(op_map)

    trt_number = len(list_of_trt_op_map)
    cuda_number = len(list_of_cuda_op_map)
    cpu_number = len(list_of_cpu_op_map)

    if debug:
        print("number of list_of_trt_op_map: {}".format(trt_number))
        print("number of list_of_cuda_op_map: {}".format(cuda_number))
        print("number of list_of_cpu_op_map: {}".format(cpu_number))

    results = []

    if trt_number == 2:
        trt_op_map = list_of_trt_op_map[0]
        trt_fp16_op_map = list_of_trt_op_map[1]

        if cuda_number > 0:
            cuda_op_map = list_of_cuda_op_map[0]
            results.append(calculate_metrics(trt_op_map, cuda_op_map))
            results.append(calculate_metrics(trt_fp16_op_map, cuda_op_map))

    elif trt_number == 1:
        trt_op_map = list_of_trt_op_map[0]

        if cuda_number > 0:
            cuda_op_map = list_of_cuda_op_map[0]
            results.append(calculate_metrics(trt_op_map, cuda_op_map))
    elif cuda_number == 1:
        cuda_op_map = list_of_cuda_op_map[0]
        calculate_metrics({}, cuda_op_map)



    '''
    trt_fall_back = False
    if trt_number > 0 and cuda_number <= 1: # TRT can execute model without falling back to CUDA/CPU
        print("Generate the metrics of TRT/TRT_fp16/CUDA ...")

        trt_op_map = list_of_trt_op_map[0]
        if trt_number > 1:
            trt_fp16_op_map = list_of_trt_op_map[1]

        if cuda_number > 0:
            cuda_op_map = list_of_cuda_op_map[0]

        results.append(calculate_metrics(trt_op_map, cuda_op_map))
        results.append(calculate_metrics(trt_fp16_op_map, cuda_op_map))

    elif trt_number > 0 and (cuda_number > 1 or cpu_number > 1): # TRT can't execute model and falling back to CUDA/CPU
        print("Generate the metrics of CUDA/CPU (Fall back due to TRT fails ...")
        trt_fall_back = True 

        trt_op_map = list_of_trt_op_map[0]
        if trt_number > 1:
            trt_fp16_op_map = list_of_trt_op_map[1]

        if cuda_number > 0:
            cuda_op_map = list_of_cuda_op_map[-1] 
        if cpu_number > 0 :
            cpu_op_map = list_of_cpu_op_map[-1]

        results.append(calculate_metrics(trt_op_map, cuda_op_map))
        results.append(calculate_metrics(trt_fp16_op_map, cuda_op_map))
    '''

    if debug:
        print('TRT operator map:')
        pp.pprint(trt_op_map)
        print('TRT FP16 operator map:')
        pp.pprint(trt_fp16_op_map)
        print('CUDA operator map:')
        pp.pprint(cuda_op_map)
        print('CPU operator map:')
        pp.pprint(cpu_op_map)

    return results





           






