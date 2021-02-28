import subprocess
import json
import pprint
import logging
import coloredlogs
import re
import sys

debug = False
debug_verbose = False 

def get_output(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output = p.stdout.decode("ascii").strip()
    return output

def find(regex_string): 
    import glob
    results = glob.glob(regex_string)
    results.sort()
    return results

def pretty_print(pp, json_object):
    pp.pprint(json_object)
    sys.stdout.flush()

def get_latest_commit_hash():
    commit = get_output(["git", "rev-parse", "--short", "HEAD"])
    return commit

def parse_single_file(f):

    try:
        data = json.load(f)
    except Exception as e:
        return None

    model_run_flag = False
    first_run_flag = True
    provider_op_map = {}  # ep -> map of operator to duration
    provider_op_map_first_run = {} # ep -> map of operator to duration

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
        pprint._sorted = lambda x:x
        pprint.sorted = lambda x, key=None: x
        pp = pprint.PrettyPrinter(indent=4)
        print("------First run ops map (START)------")
        for key, map in provider_op_map_first_run.items():
            print(key) 
            pp.pprint({k: v for k, v in sorted(map.items(), key=lambda item: item[1], reverse=True)})

        print("------First run ops map (END) ------")
        print("------Second run ops map (START)------")
        for key, map in provider_op_map.items():
            print(key) 
            pp.pprint({k: v for k, v in sorted(map.items(), key=lambda item: item[1], reverse=True)})
        print("------Second run ops map (END) ------")

    if model_run_flag:
        return provider_op_map

    return None

def calculate_cuda_op_percentage(cuda_op_map):
    if not cuda_op_map or len(cuda_op_map) == 0:
        return 0

    cuda_ops = 0
    cpu_ops = 0
    for key, value in cuda_op_map.items():
        if key == 'CUDAExecutionProvider':
            cuda_ops += len(value)

        if key == 'CPUExecutionProvider':
            cpu_ops += len(value)

    return cuda_ops / (cuda_ops + cpu_ops)

##########################################
# Return: total ops executed in TRT,
#         total ops,
#         ratio of ops executed in TRT,
##########################################
def calculate_trt_op_percentage(trt_op_map, cuda_op_map):
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

    return ((total_ops - total_cuda_and_cpu_ops), total_ops, ratio_of_ops_in_trt)

def get_total_ops(op_map):
    total_ops = 0

    for ep in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        if ep in op_map:
            total_ops += len(op_map[ep])

    return total_ops


##########################################
# Return: total TRT execution time,
#         total execution time,
#         ratio of execution time in TRT
##########################################
def calculate_trt_latency_percentage(trt_op_map):
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

    return (total_trt_execution_time, total_execution_time, ratio_of_trt_execution_time)



def get_profile_metrics(path, profile_already_parsed, logger=None):
    logger.info("Parsing/Analyzing profiling files in {} ...".format(path))
    p1 = subprocess.Popen(["find", path, "-name", "onnxruntime_profile*", "-printf", "%T+\t%p\n"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    profiling_files = stdout.split("\n")
    logger.info(profiling_files)

    data = []
    for profile in profiling_files:
        profile = profile.split('\t')[1]
        if profile in profile_already_parsed:
            continue
        profile_already_parsed.add(profile)

        logger.info("start to parse {} ...".format(profile))
        with open(profile) as f:
            op_map = parse_single_file(f)
            if op_map:
                data.append(op_map)

    if len(data) == 0:
        logger.info("No profile metrics got.")
        return None

    return data[-1]

