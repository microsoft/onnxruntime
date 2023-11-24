#!/usr/bin/python

import argparse
import fnmatch
import json
import subprocess as sp
from collections import defaultdict

import pandas as pd


def _demangle(name, demangler="c++filt"):
    try:
        with sp.Popen([demangler, name], stdin=sp.PIPE, stdout=sp.PIPE) as proc:
            out, _ = proc.communicate()
            return out.decode("utf-8").strip()
    except Exception:
        return name


def _get_args():
    parser = argparse.ArgumentParser(description="onnxruntime bench tool")
    parser.add_argument("input", type=str, help="Trace input file, formatted as JSON")
    parser.add_argument(
        "--demangler",
        required=False,
        type=str,
        default="c++filt",
        help="The command to use to demangle C++ identifiers",
    )
    parser.add_argument(
        "--shape-sensitive",
        action="store_true",
        help="Perform a shape sensitive analysis of kernel execution times",
    )

    parser.add_argument(
        "--dimension-sensitive",
        action="store_true",
        help="Perform a kernel launch dimension sensitive analysis of kernel execution times",
    )

    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        action="extend",
        help="Restrict analysis to the specified identifiers, i.e., specify a filter list. Also supports UNIX-style wildcards.",
    )
    parser.add_argument("--csv", help="Save data to csv")
    parser.add_argument("-c", "--count", type=int, default=40, help="List top N items")

    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=1,
        help="Index of the first model run to process (starting from 0, supports negative indices). "
        "Defaults to 1 to skip the first run (run 0), which is often a warmup step.",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=None,
        help="Index of the last model run to process (exclusive, supports negative indices). "
        "Defaults to None, which means all runs starting from --start will be included.",
    )
    parser.add_argument(
        "--mapping",
        "-m",
        action="store_true",
        help="Whether dump op-kernel correlation",
    )

    args = parser.parse_args()
    return args


def _shape_to_string(shape):
    res = ""
    for dict_obj in shape:
        if len(dict_obj) > 1:
            raise ValueError("Unhandled type in _shape_to_string()")
        key = next(iter(dict_obj.keys()))
        value = next(iter(dict_obj.values()))
        if len(res) != 0:
            res += ","
        res += f'{key}({"x".join(str(v) for v in value)})'
    return res


def _json_to_df(data, filter_matcher):
    cpu_entries = []
    gpu_entries = []

    most_recent_kernel_launch_event = None
    num_missing_kernel_launch_events = 0
    total_kernel_events = 0

    for item in data:
        cat = item.get("cat")
        if cat is None:
            continue
        dur = item.get("dur")
        if dur is None:
            continue
        arg = item.get("args")
        if arg is None:
            continue
        op_name = arg.get("op_name")

        name = item["name"]

        if not filter_matcher(name) and op_name is not None and not filter_matcher(op_name):
            continue

        if cat != "Kernel" and not name.endswith("kernel_time"):
            continue
        if name.endswith("kernel_time"):
            most_recent_kernel_launch_event = item

        block_x = arg.get("block_x", -1)
        block_y = arg.get("block_y", -1)
        block_z = arg.get("block_z", -1)
        grid_x = arg.get("grid_x", -1)
        grid_y = arg.get("grid_y", -1)
        grid_z = arg.get("grid_z", -1)

        if cat == "Kernel":
            gpu_entries.append(
                {
                    "name": name,
                    "duration": dur,
                    "dimensions": f"b{block_x}x{block_y}x{block_z},g{grid_x}x{grid_y}x{grid_z}",
                    "op_name": op_name,
                    "input_type_shape": (
                        _shape_to_string(most_recent_kernel_launch_event["args"]["input_type_shape"])
                        if most_recent_kernel_launch_event is not None
                        else "unknown"
                    ),
                }
            )
            total_kernel_events += 1
            if gpu_entries[-1]["input_type_shape"] == "unknown" and "hipMem" not in gpu_entries[-1]["name"]:
                num_missing_kernel_launch_events += 1
        else:
            cpu_entries.append(
                {
                    "name": item["args"]["op_name"],
                    "duration": dur,
                    "input_type_shape": _shape_to_string(item["args"]["input_type_shape"]),
                    "output_type_shape": _shape_to_string(item["args"]["output_type_shape"]),
                }
            )

    if num_missing_kernel_launch_events > 0:
        print(
            f"WARNING: Could not resolve shapes for {num_missing_kernel_launch_events} of {total_kernel_events} kernels."
        )

    cpu_df = pd.DataFrame(cpu_entries)
    gpu_df = pd.DataFrame(gpu_entries)
    cpu_df["count"] = 1
    gpu_df["count"] = 1
    return cpu_df, gpu_df


def _print_top_hitters(frame, args, target="cpu"):
    if len(frame) == 0:
        print(f"No {target.upper()} entries found!")
        return
    top = args.count
    group_key = ["name"]

    if target.lower() == "gpu" and args.dimension_sensitive:
        group_key.append("dimensions")

    if args.shape_sensitive:
        group_key.append("input_type_shape")

    frame2 = frame[["duration", "count"]].sum()
    frame["pct"] = 100 * (frame["duration"] / frame2["duration"])
    fields = [*group_key, "duration", "pct", "count"]
    frame1 = frame[fields].groupby(group_key).sum().reset_index()
    frame1 = frame1.sort_values(by="duration", ascending=False)[:top]
    frame1["cumulative_pct"] = frame1["pct"].cumsum()
    frame1["cumulative_dur"] = frame1["duration"].cumsum()

    if target.lower() == "gpu":
        frame1["name"] = frame1["name"].apply(lambda x: _demangle(x, args.demangler))

    print(f"\n------ Top {target.upper()} Kernel Times ------")
    print(frame1.round(2).to_string(index=False))
    if args.csv:
        frame1.to_csv(f"{args.csv}_{target}_kernel_times.csv", index=False)


def _print_op_kernel_mapping_info(cpu_df, gpu_df, num_runs, csv=None):
    # Count op occurrences in the selected runs
    op_counts = defaultdict(int)
    for op in cpu_df.T.to_dict().values():
        identifiers = tuple([op["name"], op["input_type_shape"]])
        op_counts[identifiers] += 1

    # Collect kernel stats: count/duration
    stat_dict = defaultdict(lambda: defaultdict(float))
    for kernel in gpu_df.T.to_dict().values():
        op_name = kernel["op_name"]
        if op_name is None:  # Only interested in op related kernels
            continue
        input_type_shape = kernel["input_type_shape"]
        kernel_name = kernel["name"]
        dimensions = kernel["dimensions"]
        identifiers = tuple([op_name, input_type_shape, kernel_name, dimensions])
        stat_dict[identifiers]["count"] += 1
        stat_dict[identifiers]["duration"] += kernel["duration"]

    # Create the DataFrame for kernel entries with op correlation info
    kernel_list = []
    for identifiers, stat in stat_dict.items():
        op_name, input_type_shape, kernel_name, dimensions = identifiers
        op_count = op_counts.get(tuple([op_name, input_type_shape]))
        if op_count is None:
            continue
        kernel_list.append(
            {
                "op_name": op_name,
                "input_type_shape": input_type_shape,
                "op_count": op_count / num_runs,  # Average op count per run
                "kernel_name": kernel_name,
                "kernel_dimensions": dimensions,
                "kernel_count": stat["count"] / num_runs,  # Average kernel count per run
                "kernel_avg_dur (us)": stat["duration"] / stat["count"],
                "kernel_total_dur (us)": stat["duration"] / num_runs,
            }
        )

    df = pd.DataFrame(kernel_list)
    df["op_dur (us)"] = df.groupby(["op_name", "input_type_shape"])["kernel_total_dur (us)"].transform("sum")
    df["op_avg_dur (us)"] = df["op_dur (us)"] / df["op_count"]
    df = df.sort_values(
        by=["op_dur (us)", "op_name", "input_type_shape", "kernel_total_dur (us)"],
        ascending=False,
    ).reset_index(drop=True)
    df["kernel_pct (%)"] = df["kernel_total_dur (us)"] / df["op_dur (us)"] * 100
    df["op_pct (%)"] = df["op_dur (us)"] / df["kernel_total_dur (us)"].sum() * 100
    # Move kernel_name to the end since it tends to be long
    df.insert(len(df.columns) - 1, "kernel_name", df.pop("kernel_name"))
    if csv is not None:
        df.to_csv(f"{csv}_op_kernel_mapping.csv", index=False)


def _construct_filter_matcher(args):
    if args.filter is None or len(args.filter) == 0:
        return lambda x: True
    filter_list = args.filter
    concrete_filter_set = set()
    fnmatch_filter_set = set()
    for pattern in filter_list:
        if "*" in pattern or "?" in pattern or "[" in pattern or "]" in pattern:
            fnmatch_filter_set.add(pattern)
        else:
            concrete_filter_set.add(pattern)

    def _match_item(item):
        if item in concrete_filter_set:
            return True
        return any(fnmatch.fnmatch(item, pattern) for pattern in fnmatch_filter_set)

    return _match_item


def _split_data_across_runs(data, start=1, end=None):
    """
    Splits the traces according to model runs they belong to.
    By default, we skip the first model run (run 0) and consider all subsequent runs.
    """
    # Here we assume that the traces are properly ordered, so we can simplify the splitting logic.
    model_run_splits = [i for i, item in enumerate(data) if item.get("name") == "model_run"]
    if not model_run_splits:
        print('WARNING: Could not find "model_run" event in trace. Using entire traces.')
        return data
    total_num_runs = len(model_run_splits)
    print(f"Found {total_num_runs} model_run events in trace.")

    assert -total_num_runs <= start < total_num_runs, f"Invalid start index {start}."
    if start < 0:
        start += total_num_runs
    if end is None:
        end = total_num_runs
    else:
        assert -total_num_runs <= end < total_num_runs, f"Invalid end index {end}."
        if end < 0:
            end += total_num_runs
    num_runs = end - start
    assert num_runs > 0, "No valid model runs are included in the split."
    print(f"Analyzing {num_runs} model run(s): {start}-{end - 1}.")

    # Add index 0 in case user wants to include the first model run.
    model_run_splits = [0, *model_run_splits]
    return data[model_run_splits[start] : model_run_splits[end]], num_runs


def _load_json(profile_path):
    with open(profile_path, encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if isinstance(data, dict):
        data = data["traceEvents"]
    return data


def main():
    args = _get_args()
    filter_matcher = _construct_filter_matcher(args)

    data = _load_json(args.input)
    data, num_runs = _split_data_across_runs(data, args.start, args.end)
    cpu_df, gpu_df = _json_to_df(data, filter_matcher)

    pd.set_option("display.max_colwidth", 120)
    _print_top_hitters(cpu_df, args, target="cpu")
    _print_top_hitters(gpu_df, args, target="gpu")
    if args.mapping:
        _print_op_kernel_mapping_info(cpu_df, gpu_df, num_runs, args.csv)


if __name__ == "__main__":
    main()
