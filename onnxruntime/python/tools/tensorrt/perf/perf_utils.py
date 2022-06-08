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


def find_files(path, name_pattern, are_dirs=False):
    files = []

    cmd = ["find", path, "-name", name_pattern]
    if are_dirs:
        cmd += ["-type", "d"]

    files_str = get_output(cmd)
    if files_str:
        files = files_str.split("\n")

    return files

def find(regex_string):
    import glob

    results = glob.glob(regex_string)
    results.sort()
    return results


def pretty_print(pp, json_object):
    pp.pprint(json_object)
    sys.stdout.flush()

def parse_model_run_info(data):
    model_run_info = []

    for entry in data:
        if entry["cat"] == "Session" and entry["name"] == "model_run":
            model_run_info.append(entry)

    return model_run_info


def add_op_map_entry(provider_op_map, provider, op_name, duration):
    if provider not in provider_op_map:
        provider_op_map[provider] = {}

    op_map = provider_op_map[provider]

    if op_name not in op_map:
        op_map[op_name] = {
            "num_instances": 1,
            "total_dur": duration,
            "min_dur": duration,
            "max_dur": duration
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

    :return: The parsed operator usage information.
    """

    provider_node_op_map = {}  # ep -> map of node operator info
    provider_kernel_op_map = {}  # ep -> map of kernel operator info
    model_run_start = target_model_run["ts"]
    model_run_end = model_run_start + target_model_run["dur"]

    # Used to track the previous CPU node that launched kernel(s).
    prev_node = None

    for entry in profile_entries:
        entry_start = entry["ts"]
        entry_end = entry_start + entry["dur"]

        # Skip entries that end before the target model run.
        if entry_end < model_run_start:
            prev_node = None
            continue

        # Stop if we encounter entries that start after the target model run ends.
        if entry_start > model_run_end:
            break

        assert (entry_start >= model_run_start) and (entry_end <= model_run_end)

        if (not "cat" in entry) or (not "name" in entry) or (not "args" in entry) or (not "op_name" in entry["args"]):
            prev_node = None
            continue

        # Parse a graph node. The node's duration represents execution time on a CPU thread (regardless of EP).
        if entry["cat"] == "Node":
            prev_node = None

            if re.search(".*kernel_time", entry["name"]) and ("provider" in entry["args"]):
                entry_args = entry["args"]
                add_op_map_entry(provider_node_op_map, entry_args["provider"], entry_args["op_name"], entry["dur"])
                prev_node = entry

        # Parse a GPU kernel that was launched by a previous node. Kernels only run on TensorRT or CUDA EPs.
        elif entry["cat"] == "Kernel" and prev_node is not None:
            add_op_map_entry(provider_kernel_op_map, prev_node["args"]["provider"], entry["args"]["op_name"],
                             entry["dur"])
        else:
            prev_node = None

    return (provider_node_op_map, provider_kernel_op_map)


def parse_session_profile(profile):
    """
    Parses a JSON profile file and returns information on operator usage per EP.

    :param profile: The file handle for the profile to parse.

    :return: Dictionary containing operator usage information per EP.
    """

    try:
        profile_entries = json.load(profile)
    except Exception:
        return None

    # Get information on where each model run starts and ends.
    model_run_info = parse_model_run_info(profile_entries)
    print("{} contains {} model runs".format(profile, len(model_run_info)))
    print(model_run_info)

    if not model_run_info:
        return None

    # Use the model run with the lowest total duration.
    min_run = min(model_run_info, key=lambda entry: entry["dur"])
    print("{} has min run with duration {}".format(profile, min_run["dur"]))

    # Parse model run
    print("Parsing model run for {}".format(profile))
    op_maps = parse_model_run(profile_entries, min_run)

    print(op_maps)
    return op_maps


def get_profile_metrics(path, profile_already_parsed, profile_file_prefix, logger=None):
    logger.info("Parsing/Analyzing profiling files in {} ...".format(path))
    p1 = subprocess.Popen(
        ["find", path, "-name", f"{profile_file_prefix}*", "-printf", "%T+\t%p\n"],
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
            op_maps = parse_session_profile(f)
            if op_maps and op_maps[0]:
                data.append(op_maps)

    if len(data) == 0:
        logger.info("No profile metrics got.")
        return None

    return data[-1]
