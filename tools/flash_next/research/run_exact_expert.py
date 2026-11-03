import argparse
import json
from pathlib import Path
import re
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 7), required=True)
    parser.add_argument("--columns", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--decode-only", action="store_true")
    args = parser.parse_args()
    prefix = ROOT / f"exact-expert-v2-c{args.columns}-gpu{args.gpu}-20261029"
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    directory = Path("/home/sayanshaw/repos/qwen38-int4-exact-expert")
    manifest = json.loads((directory / "fixtures.json").read_text())
    if args.decode_only:
        manifest = [fixture for fixture in manifest if fixture["rows"] == 1]
    build = json.loads((directory / "build-v2/build-manifest.json").read_text())
    assert sha(directory / "build-v2/libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    fixtures = " ".join("/workspace/qwen38-int4-exact-expert/fixtures/" + f["fixture"] for f in manifest)
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        + gpu_uuid + "; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_HC_EXACT_DENSE=1 "
        "ORT_WEBGPU_EXACT_DENSE_MID=1 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
        "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 "
        f"EXPERT_COLUMNS={args.columns} HC_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile; "
        "exec /workspace/qwen38-int4-exact-expert/build-v2/native_compare "
        "/workspace/qwen38-int4-exact-expert/build-v2/libonnxruntime_providers_webgpu.so " + fixtures)
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", str(args.gpu),
         "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "600",
         "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    log = prefix.with_suffix(".log").read_text()
    passes = re.findall(r"^PASS bitwise (\S+) checked_elements=(\d+)", log, re.M)
    assert len(passes) == len(manifest), (len(passes), len(manifest))
    profiles = [Path(line.split(" ", 1)[1].replace("/evidence/", str(ROOT.parent) + "/"))
                for line in log.splitlines() if line.startswith("PROFILE ")]
    dispatches = sum(event.get("cat") == "Kernel" and event["name"].endswith("&ExactExpert")
                     for profile in profiles for event in json.loads(profile.read_text()))
    assert dispatches == 6 * 2 * 8, dispatches
    report = dict(status="real_weight_qmoe_finite_bitwise_qualified",
                  columns=args.columns, fixtures=len(passes),
                  checked_elements_per_run=sum(int(n) for _, n in passes),
                  candidate_repetitions=8, provider_sha256=build["provider_sha256"],
                  actual_candidate_dispatches=dispatches,
                  gpu_uuid=gpu_uuid, quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
