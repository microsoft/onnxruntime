import subprocess
import json
import pprint
# import os

debug = False 

def parse_file(f):
    import re
    data = json.load(f)

    model_run_flag = False 
    first_run_flag = True
    provider_op_map = {}  # ep -> map of operators and duration
    provider_op_map_first_run = {} # ep -> of operators and duration

    for row in data:
        if not "cat" in row:
            continue

        if row["cat"] == "Session":
            if "name" in row and row["name"] == "model_run":
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


    if debug:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(provider_op_map_first_run)
        pp.pprint(provider_op_map)

    if model_run_flag:
        return provider_op_map 
    
    return None


#
# Return: total ops executed in TRT,
#         total ops,
#         ratio of ops executed in TRT,
#         ratio of execution time in TRT
#
#
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
            op_map = parse_file(f)
            if op_map:
                data.append(op_map)

    # pp.pprint(data)
    trt_op_map = {}
    cuda_op_map = {}

    for item in data:
        if "TensorrtExecutionProvider" in item:
            trt_op_map = item
        elif  "CUDAExecutionProvider" in item:
            cuda_op_map = item
    if debug:
        pp.pprint(trt_op_map)
        pp.pprint(cuda_op_map)


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

    if len(trt_op_map) == 0:
        total_cuda_and_cpu_ops = total_ops

    ratio_of_trt = (total_ops - total_cuda_and_cpu_ops) / total_ops
    print(total_cuda_and_cpu_ops)
    print(total_ops)
    print(ratio_of_trt)


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
    print(total_trt_execution_time)
    print(total_execution_time)
    print(ratio_of_trt_execution_time)

    return (total_ops - total_cuda_and_cpu_ops), total_ops, ratio_of_trt, ratio_of_trt_execution_time

           






