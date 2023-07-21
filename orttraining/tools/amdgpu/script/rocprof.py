import argparse
import csv
import os  # noqa: F401

import numpy as np  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()


def get_gpu_lines(path):
    lines = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row[2].find("TotalDurationNs") < 0:
                lines.append(row)  # noqa: PERF401
        return lines


activities = [
    ("nccl", lambda x: x.find("nccl") >= 0),
    ("gemm", lambda x: x.find("Cijk_") >= 0),
    ("memcpy", lambda x: x.find("CUDA mem") >= 0),
    ("adam", lambda x: x.lower().find("adam") >= 0),
    ("lamb", lambda x: x.lower().find("lamb") >= 0 or x.lower().find("multi_tensor_apply") >= 0),
    ("dropout", lambda x: x.lower().find("dropout") >= 0 or x.find("curand") >= 0),
    ("layernorm", lambda x: x.find("LayerNorm") >= 0 or x.find("cuCompute") >= 0),
    ("reduce", lambda x: x.find("reduce") >= 0),
    ("softmax", lambda x: x.lower().find("softmax") >= 0),
    ("transpose", lambda x: x.lower().find("transpose") >= 0),
    ("element-wise", lambda x: x.lower().find("elementwise") >= 0 or x.find("DivGrad") >= 0),
    ("jit", lambda x: x.startswith("kernel_")),
    ("misc", lambda x: True),
]


def group_gpu_activity(lines):
    groups = {name: [] for name, _ in activities}
    for line in lines:
        for name, check in activities:
            if check(line[0]):
                groups[name].append(line)
                break
    return groups


def get_seconds(time):
    return float(time.replace("us", "")) / (1000.0 * 1000.0 * 1000.0)


def gpu_percent_time(activities):
    return sum([float(a[4].replace("%", "")) for a in activities])


def gpu_absolute_time(activities):
    return sum([get_seconds(a[2]) for a in activities])


def gpu_kernel_calls(activities):
    return sum([int(a[1]) for a in activities])


lines = get_gpu_lines(args.input)
groups = group_gpu_activity(lines)

for name in groups:
    activities = groups[name]
    print(
        "{}: N={}, calls={}, absolute={:.3f}s, percent={:.2f}%".format(
            name,
            len(activities),
            gpu_kernel_calls(activities),
            gpu_absolute_time(activities),
            gpu_percent_time(activities),
        )
    )

total = [item for name in groups for item in groups[name]]
print(
    "Total: N={}, calls={}, absolute={:.3f}s, percent={:.2f}%".format(
        len(total), gpu_kernel_calls(total), gpu_absolute_time(total), gpu_percent_time(total)
    )
)
