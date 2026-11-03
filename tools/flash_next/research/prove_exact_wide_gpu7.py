import argparse
import collections
import json
from pathlib import Path
import statistics
import subprocess

from qualify_quickgelu_mul_long_context import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = ROOT / ("exact-wide-filtered-gpu7-route-20261028" if args.filtered
                     else "exact-wide-gpu7-route-20261028")
    profile_prefix = ("exact-wide-filtered-route-profile-20261028" if args.filtered
                      else "exact-wide-route-profile-20261028")
    if args.output_prefix:
        prefix = args.output_prefix
        profile_prefix = prefix.name + "-profile"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    directory = Path("/home/sayanshaw/repos/qwen38-int4-exact-wide-gpu7")
    fixtures = json.loads((directory / "fixtures.json").read_text())
    shapes = sorted({(f["K"], f["N"]) for f in fixtures if f["kind"] == "actual"})
    selected = [next(f for f in fixtures if (f["K"], f["N"]) == shape) for shape in shapes]
    paths = " ".join("/workspace/qwen38-int4-exact-wide-gpu7/fixtures/" + f["fixture"] for f in selected)
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; "
        "export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_HC_EXACT_DENSE=1 "
        "ORT_WEBGPU_EXACT_DENSE_MID=1 "
        "HC_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/exact-wide-route-profile-20261028; "
        "exec /workspace/qwen38-int4-exact-wide-gpu7/build/native_compare "
        "/workspace/qwen38-int4-exact-wide-gpu7/build/libonnxruntime_providers_webgpu.so " + paths
    )
    command = command.replace("exact-wide-route-profile-20261028", profile_prefix)
    if args.filtered:
        command = command.replace("/workspace/qwen38-int4-exact-wide-gpu7/build/",
                                  "/workspace/qwen38-int4-exact-wide-gpu7-filtered/build/")
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "180",
                    "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    profiles = sorted(ROOT.glob(profile_prefix + "-*_*.json"))
    assert len(profiles) == len(selected), (len(profiles), len(selected))
    grouped = collections.defaultdict(list)
    count = collections.Counter()
    for path in profiles:
        for event in json.loads(path.read_text()):
            if event.get("cat") == "Kernel":
                program = event["name"].split("&")[-1]
                count[program] += 1
                grouped[(program, event["args"]["shapes"])].append(event["dur"])
    expected_new = sum(not args.filtered or k != 2560 or n == 1 for k, n in shapes)
    assert count["ExactDenseWide"] == expected_new * 8, count
    assert count["MatMulNBits"] == len(selected) * 9 + (len(selected) - expected_new) * 8, count
    report = dict(status="actual_exact_wide_gpu_dispatch_proven",
                  actual_weight_shapes=shapes, actual_kernel_counts=dict(count),
                  timings=[dict(program=key[0], shapes=key[1], samples=len(values),
                                median_profiled_gpu_us=statistics.median(values))
                           for key, values in sorted(grouped.items())],
                  profiles=[str(p) for p in profiles],
                  scope="Instrumented fixture execution, not application TPS or a production throughput claim.")
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
