# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import re
import subprocess
import sys

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
op_metrics_name = "op_metrics"
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
    ("input_ep", "Input EP", "InputEP"),
    ("operator", "Operator", "Operator"),
    ("assigned_ep", "Assigned EP", "AssignedEP"),
    ("event_category", "Event Category", "EventCategory"),
    ("num_instances", "Num Instances", "NumInstances"),
    ("total_dur", "Total Duration", "TotalDuration"),
    ("min_dur", "Min Duration", "MinDuration"),
    ("max_dur", "Max Duration", "MaxDuration"),
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


def split_and_sort_output(string_list):
    string_list = string_list.split("\n")
    string_list.sort()
    return string_list


def find_files(path, name_pattern, are_dirs=False):
    """
    Finds files that match the given name pattern within the given path.

    :param path: The path in which to search for files.
    :param name_pattern: Glob pattern (e.g., *.py) used to search for files.
    :param are_dirs: True if function should find directories instead of regular files.

    :return: A list of the found file path names.
    """

    files = []

    cmd = ["find", path, "-name", name_pattern]
    if are_dirs:
        cmd += ["-type", "d"]

    files_str = get_output(cmd)
    if files_str:
        files = files_str.split("\n")

    return files


def get_cuda_version():
    nvidia_strings = get_output(["nvidia-smi"])
    version = re.search(r"CUDA Version: \d\d\.\d", nvidia_strings).group(0)
    return version


def get_trt_version(workspace):
    libnvinfer = get_output(["find", workspace, "-name", "libnvinfer.so.*"])
    nvinfer = re.search(r".*libnvinfer.so.*", libnvinfer).group(0)
    trt_strings = get_output(["nm", "-D", nvinfer])
    version = re.search(r"tensorrt_version.*", trt_strings).group(0)
    return version


def get_linux_distro():
    linux_strings = get_output(["cat", "/etc/os-release"])
    stdout = linux_strings.split("\n")[:2]
    infos = []
    for row in stdout:
        row = re.sub("=", ":  ", row)
        row = re.sub('"', "", row)
        infos.append(row)
    return infos


def get_memory_info():
    mem_strings = get_output(["cat", "/proc/meminfo"])
    stdout = mem_strings.split("\n")
    infos = []
    for row in stdout:
        if "Mem" in row:
            row = re.sub(": +", ":  ", row)
            infos.append(row)
    return infos


def get_cpu_info():
    cpu_strings = get_output(["lscpu"])
    stdout = cpu_strings.split("\n")
    infos = []
    for row in stdout:
        if "mode" in row or "Arch" in row or "name" in row:
            row = re.sub(": +", ":  ", row)
            infos.append(row)
    return infos


def get_gpu_info():
    info = get_output(["lspci", "-v"])
    infos = re.findall("NVIDIA.*", info)
    return infos


def get_cudnn_version(workspace):
    cudnn_path = get_output(["whereis", "cudnn_version.h"])
    cudnn_path = re.search(": (.*)", cudnn_path).group(1)
    cudnn_outputs = get_output(["cat", cudnn_path])
    major = re.search("CUDNN_MAJOR (.*)", cudnn_outputs).group(1)
    minor = re.search("CUDNN_MINOR (.*)", cudnn_outputs).group(1)
    patch = re.search("CUDNN_PATCHLEVEL (.*)", cudnn_outputs).group(1)
    cudnn_version = major + "." + minor + "." + patch
    return cudnn_version


def get_system_info(root_dir):
    info = {}
    info["cuda"] = get_cuda_version()
    info["trt"] = get_trt_version(root_dir)
    info["cudnn"] = get_cudnn_version(root_dir)
    info["linux_distro"] = get_linux_distro()
    info["cpu_info"] = get_cpu_info()
    info["gpu_info"] = get_gpu_info()
    info["memory"] = get_memory_info()

    return info


def get_profile_model_runs(profile_entries):
    """
    Parses in-memory session profile data and returns all 'model run' entries.

    :param profile_entries: A list of session profile entries.

    :return: A list of model run entries.
    """

    model_run_info = []

    for entry in profile_entries:
        if entry["cat"] == "Session" and entry["name"] == "model_run":
            model_run_info.append(entry)

    return model_run_info


def add_op_map_entry(provider_op_map, provider, op_name, duration):
    """
    Adds an operator usage data point to a dictionary that tracks operator usage per EP.

    :param provider_op_map: Dictionary that tracks operator usage per EP.
    :param provider: The EP for which to add a new operator usage data point.
    :param op_name: The name of the operator.
    :param duration: The execution duration (in microseconds) of the operator.
    """

    if provider not in provider_op_map:
        provider_op_map[provider] = {}

    op_map = provider_op_map[provider]

    if op_name not in op_map:
        op_map[op_name] = {
            "num_instances": 1,
            "total_dur": duration,
            "min_dur": duration,
            "max_dur": duration,
            "subgraph": {},
        }
    else:
        op_info = op_map[op_name]

        op_info["num_instances"] += 1
        op_info["total_dur"] += duration
        op_info["min_dur"] = min(duration, op_info["min_dur"])
        op_info["max_dur"] = max(duration, op_info["max_dur"])


def parse_model_run(profile_entries, target_model_run):
    """
    Parses profile data to obtain operator usage information for the given 'model run'.

    :param profile_entries: List of profile data entries.
    :param target_model_run: Time range information on the model run to parse.

    :return: A tuple containing the parsed operator usage information for CPU nodes and GPU kernels.
    """

    provider_node_op_map = {}  # ep -> map of node operator info
    provider_kernel_op_map = {}  # ep -> map of kernel operator info
    model_run_start = target_model_run["ts"]
    model_run_end = model_run_start + target_model_run["dur"]

    # Used to track the previous CPU node that launched kernel(s).
    prev_node = {}

    for entry in profile_entries:
        entry_start = entry["ts"]
        entry_end = entry_start + entry["dur"]

        # Skip entries that end before the target model run.
        if entry_end < model_run_start:
            prev_node = {}
            continue

        # Stop if we encounter entries that start after the target model run ends.
        if entry_start > model_run_end:
            break

        assert (entry_start >= model_run_start) and (entry_end <= model_run_end)

        if (not "cat" in entry) or (not "name" in entry) or (not "args" in entry) or (not "op_name" in entry["args"]):
            prev_node = {}
            continue

        # Parse a graph node. The node's duration represents execution time on a CPU thread (regardless of EP).
        if entry["cat"] == "Node":
            prev_node = {}

            if re.search(".*kernel_time", entry["name"]) and ("provider" in entry["args"]):
                entry_args = entry["args"]
                add_op_map_entry(provider_node_op_map, entry_args["provider"], entry_args["op_name"], entry["dur"])
                prev_node = entry

        # Parse a GPU kernel that was launched by a previous node. Kernels only run on TensorRT or CUDA EPs.
        elif entry["cat"] == "Kernel" and prev_node:
            add_op_map_entry(
                provider_kernel_op_map, prev_node["args"]["provider"], entry["args"]["op_name"], entry["dur"]
            )
        else:
            prev_node = {}

    return (provider_node_op_map, provider_kernel_op_map)


def parse_session_profile(profile):
    """
    Parses a JSON profile file and returns information on operator usage per EP.

    :param profile: The file handle for the profile to parse.

    :return: A tuple containing the parsed operator usage information for CPU nodes and GPU kernels.
    """

    try:
        profile_entries = json.load(profile)
    except Exception:
        return None

    # Get information on where each model run starts and ends.
    model_runs = get_profile_model_runs(profile_entries)

    if not model_runs:
        return None

    # Use the model run with the lowest total duration.
    min_run = min(model_runs, key=lambda entry: entry["dur"])

    # Parse model run
    op_maps = parse_model_run(profile_entries, min_run)

    return op_maps


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
        profile = profile.split("\t")[1]

        logger.debug("Parsing profile %s ...", profile)
        with open(profile, encoding="utf-8") as file_handle:
            op_maps = parse_session_profile(file_handle)
            if op_maps and op_maps[0]:
                data.append(op_maps)

    if len(data) == 0:
        logger.debug("No profile metrics found.")
        return None

    return data[-1]
