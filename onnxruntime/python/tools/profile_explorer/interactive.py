import pprint
import time

import pandas as pd
from print_utils import get_cpu_top_hitters, print_frame
from utils import shape_to_string

_TOP_LEVEL_MODE = 1
_DRILL_DOWN_MODE = 2


def _get_kernel_times(cpu_df, index, args, ops_and_kernel_events):
    row = cpu_df.iloc[index]
    op_name = row["name"]
    if args.shape_sensitive:
        shape = row["input_type_shape"]
        relevant_ops_and_kernels = list(
            filter(
                lambda t: t[0]["args"]["op_name"] == op_name
                and shape_to_string(t[0]["args"]["input_type_shape"]) == shape,
                ops_and_kernel_events,
            )
        )
    else:
        shape = None
        relevant_ops_and_kernels = list(filter(lambda t: t[0]["args"]["op_name"] == op_name, ops_and_kernel_events))

    relevant_kernels = []
    for kernels in [x[1] for x in relevant_ops_and_kernels]:
        relevant_kernels.extend(kernels)
    frame = pd.DataFrame(relevant_kernels)
    frame["count"] = 1
    frame2 = frame[["duration", "count"]].sum()
    frame["pct"] = 100 * (frame["duration"] / frame2["duration"])
    group_key = ["name", "input_type_shape"] if shape is not None else ["name"]
    fields = group_key + ["duration", "pct", "count"]
    frame = frame[fields].groupby(group_key).sum().reset_index()
    frame = frame.sort_values(by="duration", ascending=False)
    frame["cumulative_pct"] = frame["pct"].cumsum()
    frame["cumulative_dur"] = frame["duration"].cumsum()
    frame.reset_index(drop=True, inplace=True)
    return (frame, shape, op_name)


def interactive_loop(cpu_df, gpu_df, data, ops_and_kernel_events, args):
    mode = _TOP_LEVEL_MODE
    drill_down_df = None
    shape = None
    op_name = None
    df = None

    while True:
        if mode == _TOP_LEVEL_MODE:
            df = get_cpu_top_hitters(cpu_df, args)
            print_frame(df, "Top CPU Ops")
            action_str = input("Enter a row ID for kernel invocation details, q to quit: ")
            if action_str.lower().startswith("q"):
                break
            if not action_str.isdigit():
                print("Invalid input!")
                time.sleep(3)
                continue
            index = int(action_str)
            if index > len(df):
                print("Index is out of bounds!")
                time.sleep(3)
                continue
            else:
                mode = _DRILL_DOWN_MODE
                drill_down_df, shape, op_name = _get_kernel_times(df, index, args, ops_and_kernel_events)

        elif mode == _DRILL_DOWN_MODE:
            print_frame(
                drill_down_df,
                f"GPU Kernel Time breakdown for op: {op_name}, shape: {shape if shape is not None else 'Any'}",
            )
            action_str = input("Press b to go back to the top-level or q to quit: ")
            if action_str.lower().startswith("b"):
                mode = _TOP_LEVEL_MODE
            elif action_str.lower().startswith("q"):
                break
            else:
                print("Invalid input!")
                time.sleep(3)
                continue
