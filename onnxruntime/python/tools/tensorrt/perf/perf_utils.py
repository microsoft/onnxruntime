import json
import logging
import pprint
import re
import subprocess
import sys

import coloredlogs

debug = False
debug_verbose = False

# ORT ep names
cpu_ep = "CPUExecutionProvider"
cuda_ep = "CUDAExecutionProvider"
trt_ep = "TensorrtExecutionProvider"
acl_ep = "ACLExecutionProvider"

# provider names
cpu = "ORT-CPUFp32"
cuda = "ORT-CUDAFp32"
cuda_fp16 = "ORT-CUDAFp16"
trt = "ORT-TRTFp32"
trt_fp16 = "ORT-TRTFp16"
standalone_trt = "TRTFp32"
standalone_trt_fp16 = "TRTFp16"
acl = "ORT-ACLFp32"

# table names
# TODO: Rename 'metrics_name' to 'op_metrics_name'
metrics_name = "metrics"
success_name = "success"
fail_name = "fail"
memory_name = "memory"
latency_name = "latency"
status_name = "status"
latency_over_time_name = "latency_over_time"
specs_name = "specs"
session_name = "session"

# column names
model_title = "Model"
group_title = "Group"

# List of column name tuples for operator metrics: (<map_key>, <csv_column>, <db_column>)
op_metrics_columns = [
    ("model_name", "Model", "Model"),
    ("ep", "Input EP", "InputEP"),
    ("num_cpu_ops", "Num CPU Ops", "NumCPUOps"),
    ("cpu_exec_time", "CPU Ops execution time", "CPUExecTime"),
    ("cpu_ops", "CPU Ops", "CPUOps"),
    ("num_cuda_ops", "Num CUDA Ops", "NumCUDAOps"),
    ("cuda_exec_time", "CUDA Ops execution time", "CUDAExecTime"),
    ("cuda_ops", "CUDA Ops", "CUDAOps"),
    ("num_trt_ops", "Num TRT Ops", "NumTRTOps"),
    ("trt_exec_time", "TRT Ops execution time", "TRTExecTime"),
    ("trt_ops", "TRT Ops", "TRTOps"),
]

# endings
second = "_second"
csv_ending = ".csv"
avg_ending = " \nmean (ms)"
percentile_ending = " \n90th percentile (ms)"
memory_ending = " \npeak memory usage (MiB)"
session_ending = " \n session creation time (s)"
second_session_ending = " \n second session creation time (s)"
ort_provider_list = [cpu, cuda, trt, cuda_fp16, trt_fp16]
provider_list = [
    cpu,
    cuda,
    trt,
    standalone_trt,
    cuda_fp16,
    trt_fp16,
    standalone_trt_fp16,
]
table_headers = [model_title] + provider_list

# graph options
disable = "disable"
basic = "basic"
extended = "extended"
enable_all = "all"


def is_standalone(ep):
    return ep == standalone_trt or ep == standalone_trt_fp16


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


def parse_single_file(f):

    try:
        data = json.load(f)
    except Exception as e:
        return None

    model_run_flag = False
    first_run_flag = True
    provider_op_map = {}  # ep -> map of operator to duration
    provider_op_map_first_run = {}  # ep -> map of operator to duration

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
        pprint._sorted = lambda x: x
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


def get_ep_op_metrics(ep, op_map):
    op_metrics = {"num_ops": 0, "exec_time": 0, "ops": "{}"}

    if ep in op_map:
        ops = op_map[ep]
        op_metrics["ops"] = json.dumps(ops)
        op_metrics["num_ops"] = len(ops)

        for _, exec_time in ops.items():
            op_metrics["exec_time"] += int(exec_time)

    return op_metrics


######################################################################################################
# Parameters: op_map: A dictionary that maps EPs to a dictionary of operator durations.
#                     EX: {'CUDAExecutionProvider': { 'op0': 200, 'op1': 100 }, 'CPUExec...': {...}}
#
# Return: A dictionary that maps an execution provider to a dictionary of operator metrics.
#         EX: {'CPUExecutionProvider' : { 'num_ops': x, 'exec_time': y, 'ops': 'ops json string'},
#              'CUDAExecutionProvider': { ... }, 'TensorrtExecutionProvider: { ... }}
######################################################################################################
def get_op_breakdown(op_map):
    cpu_op_metrics = get_ep_op_metrics(cpu_ep, op_map)
    cuda_op_metrics = get_ep_op_metrics(cuda_ep, op_map)

    # Note that the number of TensorRT ops obtained from op_map, which is built by parsing profile data,
    # is incorrect. The profile data does not breakdown the individual operators used in TRT, and instead
    # provides only the total execution time of a particular TRT subgraph.
    #
    # In order to determine the number of operators handled by TRT, we first need to obtain profile data for an
    # inference session that uses only the CUDA and CPU EPs. This CUDA/CPU profile data serves as a baseline.
    # Then, the number of ops handled by TRT is calculated as follows:
    #
    # num_trt_ops = (baseline number of cuda/cpu ops) - (number of cpu/cuda ops used in trt inference session)
    #
    # EX: ep_to_operator = {
    #                        'ORT-CUDAFp32': {'CUDAExecutionProvider': { 'op0': 200, 'op1': 100 },
    #                                         'CPUExecutionProvider': { 'op2': 10, 'op3': 300 }},
    #                        'ORT-TRTFp32': {'CUDAExecutionProvider': { 'op1': 100 },
    #                                        'CPUExecutionProvider': { 'op2' : 10 },
    #                                        'TensorrtExecutionProvider': { 'subgraph0': 400 }}
    #                      }
    #
    # num_trt_ops = 4 - 2 = 2
    #
    # See output_metrics() for the code that performs the above computations/fixups.
    trt_op_metrics = get_ep_op_metrics(trt_ep, op_map)

    return {cpu_ep: cpu_op_metrics, cuda_ep: cuda_op_metrics, trt_ep: trt_op_metrics}


def get_profile_metrics(path, profile_already_parsed, logger=None):
    logger.info("Parsing/Analyzing profiling files in {} ...".format(path))
    p1 = subprocess.Popen(
        ["find", path, "-name", "onnxruntime_profile*", "-printf", "%T+\t%p\n"],
        stdout=subprocess.PIPE,
    )
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    profiling_files = stdout.split("\n")
    logger.info(profiling_files)

    data = []
    for profile in profiling_files:
        profile = profile.split("\t")[1]
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
