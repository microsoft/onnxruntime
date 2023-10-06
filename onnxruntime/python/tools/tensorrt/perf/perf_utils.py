import json
import logging  # noqa: F401
import pprint
import re
import subprocess
import sys

import coloredlogs  # noqa: F401

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
metrics_name = "metrics"
success_name = "success"
fail_name = "fail"
memory_name = "memory"
memory_over_time_name = "memory_over_time"
latency_name = "latency"
status_name = "status"
status_over_time_name = "status_over_time"
latency_over_time_name = "latency_over_time"
specs_name = "specs"
session_name = "session"
session_over_time_name = "session_over_time"

# column names
model_title = "Model"
group_title = "Group"

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
table_headers = [model_title, *provider_list]

# graph options
disable = "disable"
basic = "basic"
extended = "extended"
enable_all = "all"


def is_benchmark_mode(running_mode):
    """
    Returns True if the script's running mode requires running benchmarks.

    :param running_mode: A string denoting the script's running mode (i.e., 'benchmark', 'validate', or 'both')

    :return: True if benchmarking is required.
    """

    return running_mode == "benchmark" or running_mode == "both"


def is_validate_mode(running_mode):
    """
    Returns True if the script's running mode requires running inference validation.

    :param running_mode: A string denoting the script's running mode (i.e., 'benchmark', 'validate', or 'both')

    :return: True if validation is required.
    """

    return running_mode == "validate" or running_mode == "both"


def is_standalone(ep):
    return ep in (standalone_trt, standalone_trt_fp16)


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
    except Exception:
        return None

    model_run_flag = False
    first_run_flag = True
    provider_op_map = {}  # ep -> map of operator to duration
    provider_op_map_first_run = {}  # ep -> map of operator to duration

    for row in data:
        if "cat" not in row:
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

                if "op_name" not in args or "provider" not in args:
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
                    if row["name"] not in op_map:
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


def calculate_cuda_op_percentage(cuda_op_map):
    if not cuda_op_map or len(cuda_op_map) == 0:
        return 0

    cuda_ops = 0
    cpu_ops = 0
    for key, value in cuda_op_map.items():
        if key == "CUDAExecutionProvider":
            cuda_ops += len(value)

        if key == "CPUExecutionProvider":
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
        print(f"total_cuda_and_cpu_ops: {total_cuda_and_cpu_ops}")
        print(f"total_ops: {total_ops}")
        print(f"ratio_of_ops_in_trt: {ratio_of_ops_in_trt}")

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
    for ep in [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if ep in trt_op_map:
            op_map = trt_op_map[ep]

            total_time = 0
            for value in op_map.values():
                total_time += int(value)

            if ep == "TensorrtExecutionProvider":
                total_trt_execution_time = total_time

            total_execution_time += total_time

    if total_execution_time == 0:
        ratio_of_trt_execution_time = 0
    else:
        ratio_of_trt_execution_time = total_trt_execution_time / total_execution_time

    if debug:
        print(f"total_trt_execution_time: {total_trt_execution_time}")
        print(f"total_execution_time: {total_execution_time}")
        print(f"ratio_of_trt_execution_time: {ratio_of_trt_execution_time}")

    return (total_trt_execution_time, total_execution_time, ratio_of_trt_execution_time)


def get_profile_metrics(path, profile_file_prefix, logger):
    """
    Parses a session profile file to obtain information on operator usage per EP.

    :param path: The path containing the session profile file.
    :param profile_file_prefix: Custom prefix for session profile names. Refer to ORT SessionOptions.
    :param logger: The logger object to use for debug/info logging.

    :return: A tuple containing the parsed operator usage information for CPU nodes and GPU kernels.
    """

    logger.debug("Parsing/Analyzing profiling files in %s ...", path)

    find_proc = subprocess.Popen(
        ["find", path, "-name", f"{profile_file_prefix}*", "-printf", "%T+\t%p\n"],
        stdout=subprocess.PIPE,
    )
    sort_proc = subprocess.Popen(["sort"], stdin=find_proc.stdout, stdout=subprocess.PIPE)
    stdout, sterr = sort_proc.communicate()
    stdout = stdout.decode("ascii").strip()
    profiling_files = stdout.split("\n")
    logger.info(profiling_files)

    data = []
    for profile in profiling_files:
        profile = profile.split("\t")[1]  # noqa: PLW2901

        logger.debug("Parsing profile %s ...", profile)
        with open(profile, encoding="utf-8") as fd:
            op_map = parse_single_file(fd)
            if op_map:
                data.append(op_map)

    if len(data) == 0:
        logger.debug("No profile metrics found.")
        return None

    return data[-1]
