import argparse
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha, snapshot


GPU_UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-exact-wide-gpu7/build")
PREFIX = ROOT / "exact-wide-gpu7-native-20261028"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=15)
    args = parser.parse_args()
    build = (Path(str(BUILD.parent) + "-filtered") / "build") if args.filtered else BUILD
    prefix = ROOT / ("exact-wide-filtered-gpu7-native-20261028" if args.filtered else PREFIX.name)
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    assert sha(build / "libonnxruntime_providers_webgpu.so") == json.loads(
        (build / "build-manifest.json").read_text())["provider_sha256"]
    initial = snapshot()
    fixtures = sorted((BUILD.parent / "fixtures").glob("*.onnx"))
    assert len(fixtures) == 309
    invocation = (
        "set -e; "
        "/workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + GPU_UUID + "; "
        "export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_HC_EXACT_DENSE=1 "
        "ORT_WEBGPU_EXACT_DENSE_MID=1; "
        "exec /workspace/qwen38-int4-exact-wide-gpu7/build/native_compare "
        "/workspace/qwen38-int4-exact-wide-gpu7/build/libonnxruntime_providers_webgpu.so "
        "/workspace/qwen38-int4-exact-wide-gpu7/fixtures/*.onnx"
    )
    if args.filtered:
        invocation = invocation.replace(
            "/workspace/qwen38-int4-exact-wide-gpu7/build/",
            "/workspace/qwen38-int4-exact-wide-gpu7-filtered/build/")
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
         "--idle-seconds", "3", "--wait-seconds", str(args.wait_seconds), "--run-seconds", "900",
         "--output-prefix", str(prefix), "--", "bash", "-lc", invocation], check=True,
    )
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert log.count("PASS bitwise ") == 309 and "FAIL" not in log and "GetCapability()" not in log
    checks = sum(int(line.split("checked_elements=")[1].split()[0])
                 for line in log.splitlines() if line.startswith("PASS bitwise "))
    output = dict(status="native_finite_bitwise_qualified_not_model_qualified",
                  fixtures=309, finite_elements_per_run=checks, runs_each=16,
                  finite_bitwise_comparisons=checks * 16,
                  initial_resources=initial, final_resources=snapshot(),
                  provider_sha256=sha(build / "libonnxruntime_providers_webgpu.so"),
                  filtered=args.filtered,
                  quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
