import json
import pprint
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


def parse_next_model_run(data, start_index):
    """
    Parses profile data to obtain operator usage information for the next 'model run'.

    :param data: List of profile data entries.
    :param start_index: Starting index of the first data entry to parse.

    :return: A tuple consisting of the parsed operator usage information and the index of the next model run.
    """

    provider_op_map = {}  # ep -> map of operator to duration
    num_entries = len(data)
    index = -1

    # Parse until the beginning of a model run.
    for idx, row in enumerate(data[start_index:]):
        if row["cat"] == "Session" and row["name"] == "model_run":
            index = idx + 1
            break

    # Exit if did not find a model run.
    if index < 0:
        return (provider_op_map, num_entries)

    prev_node = None

    # Parse nodes in the model run
    while index < num_entries:
        entry = data[index]

        # Stop once we encounter the next model run.
        if entry["cat"] == "Session" and entry["name"] == "model_run":
            break

        if (not "cat" in entry) or (not "name" in entry) or (not "args" in entry) or (not "op_name" in entry["args"]):
            prev_node = None
            index += 1
            continue

        # Parse a graph node that is assigned to some EP. The node's duration includes only CPU execution time.
        if entry["cat"] == "Node":
            prev_node = None

            if re.search(".*kernel_time", entry["name"]) and ("provider" in entry["args"]):
                args = entry["args"]
                provider = args["provider"]

                if provider not in provider_op_map:
                    provider_op_map[provider] = {}

                op_map = provider_op_map[provider]
                op_map[entry["name"]] = {"ts": entry["ts"], "dur": entry["dur"], "op_name": args["op_name"]}
                prev_node = op_map[entry["name"]]

        # Parse a CUDA kernel spawned by a previous node. The kernel's duration corresponds to GPU exec time.
        elif entry["cat"] == "Kernel" and prev_node and entry["args"]["op_name"] == prev_node["op_name"]:
            prev_node_end = prev_node["ts"] + prev_node["dur"]
            kernel_start = entry["ts"]
            kernel_end = kernel_start + entry["dur"]

            # The node runs in the CPU and the kernel runs in the GPU. So, only add the portion of the
            # kernel's duration that does not overlap with the corresponding CPU operation.
            if kernel_end > prev_node_end:
                overlap = prev_node_end - kernel_start
                prev_node["dur"] += entry["dur"] if overlap <= 0 else entry["dur"] - overlap
        else:
            prev_node = None

        index += 1

    # Return the parsed model run info and the index to the end of the run.
    return (provider_op_map, index)


def parse_single_file(f):
    """
    Parses a JSON profile file and returns information on op usage per EP.

    :param f: The string contents of the profile file.

    :return: Dictionary containing operator usage information per EP.
    """

    try:
        data = json.load(f)
    except Exception:
        return None

    # Parse first model run
    (provider_op_map, index) = parse_next_model_run(data, 0)

    # Parse and keep the second model run (if available).
    if index < len(data):
        (op_map, index) = parse_next_model_run(data, index)

        if op_map:
            provider_op_map = op_map

    return provider_op_map


def get_ep_operator_metrics(ep, ep_nodes):
    """
    Returns a dictionary of summarized operator metrics.

    :param ep: The EP for which to summarize operator metrics. Should be a key in `ep_nodes`.
    :param ep_nodes: A dictionary that maps an ORT execution provider to a dictionary of node information.

        Ex: {
                "CUDAExecutionProvider": {
                    "node0": {"dur": 200, "op_name": "Conv"},
                    "node1": {"dur": 100, "op_name": "Conv"},
                    ...
                },
                "CPUExecutionProvider": {...}
            }

    :return: A dictionary of summarized operator metrics.

        Ex: {
                "Conv": {"num_instances": 20, "total_dur": 32003, "min_dur": 10, "max_dur": 200},
                "Add": {"num_instances": 22, "total_dur": ... }
            }
    """

    ep_operator_metrics = {}

    if ep in ep_nodes:
        nodes = ep_nodes[ep]

        for _, node_info in nodes.items():
            node_dur = int(node_info["dur"])
            op_name = node_info["op_name"]

            if op_name not in ep_operator_metrics:
                ep_operator_metrics[op_name] = {
                    "num_instances": 1,
                    "total_dur": node_dur,
                    "min_dur": node_dur,
                    "max_dur": node_dur,
                }
            else:
                node_op_info = ep_operator_metrics[op_name]
                node_op_info["num_instances"] += 1
                node_op_info["total_dur"] += node_dur
                node_op_info["min_dur"] = min(node_dur, node_op_info["min_dur"])
                node_op_info["max_dur"] = max(node_dur, node_op_info["max_dur"])

    return ep_operator_metrics


def get_operator_metrics(ep_nodes):
    """
    Returns the number of operators and the total execution time for each execution provider.

    :param ep_nodes: A dictionary that maps an ORT execution provider to a dictionary of node information.

        Ex: {
                "CUDAExecutionProvider": {
                    "node0": {"dur": 200, "op_name": "Conv"},
                    "node1": {"dur": 100, "op_name": "Conv"},
                    ...
                },
                "CPUExecutionProvider": {...}
            }

    :return: A dictionary that maps an ORT execution provider to a dictionary of summarized operator metrics.

        Ex: {
                "CPUExecutionProvider" : {
                    "Conv": {"num_instances": 20, "total_dur": 32003, "min_dur": 10, "max_dur": 200},
                    "Add": {"num_instances": 22, "total_dur": ... }
                },
                "CUDAExecutionProvider": { ... },
                "TensorrtExecutionProvider: { ... }
            }
    """

    return {
        cpu_ep: get_ep_operator_metrics(cpu_ep, ep_nodes),
        cuda_ep: get_ep_operator_metrics(cuda_ep, ep_nodes),
        trt_ep: get_ep_operator_metrics(trt_ep, ep_nodes),
    }


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
