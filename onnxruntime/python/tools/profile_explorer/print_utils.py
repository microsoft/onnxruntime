from utils import demangle


def get_cpu_top_hitters(frame, args):
    if len(frame) == 0:
        print("No CPU entries found!")
        return
    top = args.count
    group_key = ["name"]
    if args.shape_sensitive:
        group_key.append("input_type_shape")

    frame2 = frame[["duration", "count"]].sum()
    frame["pct"] = 100 * (frame["duration"] / frame2["duration"])
    fields = group_key + ["duration", "pct", "count"]
    frame1 = frame[fields].groupby(group_key).sum().reset_index()
    frame1 = frame1.sort_values(by="duration", ascending=False)[:top]
    frame1["cumulative_pct"] = frame1["pct"].cumsum()
    frame1["cumulative_dur"] = frame1["duration"].cumsum()

    frame1.reset_index(drop=True, inplace=True)
    return frame1


def print_frame(frame, preamble):
    print(f"\n------ {preamble} ------")
    print(frame.round(2).to_string(index=True))
    print("--------------------------------------------------\n")


def print_cpu_top_hitters(frame, args):
    frame1 = get_cpu_top_hitters(frame, args)
    print_frame(frame1, "Top CPU Op Times")


def print_gpu_top_hitters(frame, args):
    if len(frame) == 0:
        print("No GPU entries found!")
        return
    top = args.count
    group_key = ["name"]
    if args.dimension_sensitive:
        group_key.append("dimensions")
    if args.shape_sensitive:
        group_key.append("input_type_shape")

    frame2 = frame[["duration", "count"]].sum()
    frame["pct"] = 100 * (frame["duration"] / frame2["duration"])
    fields = group_key + ["duration", "pct", "count"]
    frame1 = frame[fields].groupby(group_key).sum().reset_index()
    frame1 = frame1.sort_values(by="duration", ascending=False)[:top]
    frame1["cumulative_pct"] = frame1["pct"].cumsum()
    frame1["cumulative_dur"] = frame1["duration"].cumsum()
    frame1["name"] = frame1["name"].apply(lambda x: demangle(x, args.demangler))
    print("\n------ Top GPU Kernel Times ------")
    print(frame1.round(2).to_string(index=False))
    print("--------------------------------------------------\n")
    if args.csv:
        frame1.to_csv(f"{args.csv}_gpu_kernel_times.csv", index=False)
