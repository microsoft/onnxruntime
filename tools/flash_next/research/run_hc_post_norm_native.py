import argparse
import collections
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v3", "v4", "v5", "v6", "v7", "v8"), default="v8")
    parser.add_argument("--gpu-index", type=int, choices=(0, 3, 7), default=7)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--fixture-set", choices=("standard", "boundaries"), default="standard")
    args = parser.parse_args()
    directory = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm")
    build_dir = directory / ("build-" + args.revision)
    prefix = args.output_prefix or ROOT / ("hc-post-norm-" + args.revision + "-native-20261029")
    gpu_uuid = {0: "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd",
                3: "GPU-38eac315-6ffb-8cb1-c68f-521767e9ba3e",
                7: "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"}[args.gpu_index]
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    fixture_name = "fixtures" if args.fixture_set == "standard" else "fixtures-boundaries"
    manifest = json.loads((directory / (fixture_name + ".json")).read_text())
    build = json.loads((build_dir / "build-manifest.json").read_text())
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        f"{gpu_uuid}; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile; "
        f"exec /workspace/qwen38-int4-hc-post-norm/build-{args.revision}/native_compare "
        f"/workspace/qwen38-int4-hc-post-norm/build-{args.revision}/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-hc-post-norm/" + fixture_name + "/" +
                 f["fixture"] for f in manifest))
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", str(args.gpu_index),
         "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "180",
         "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS exact finite outputs ") == len(manifest)
    for fixture in manifest:
        stem = Path(fixture["fixture"]).stem
        profiles = list(ROOT.glob(prefix.name + "-profile-" + stem + "-fused*.json"))
        assert len(profiles) == 1, profiles
        count = collections.Counter(event["name"].split("&")[-1]
                                    for event in json.loads(profiles[0].read_text())
                                    if event.get("cat") == "Kernel")
        assert count == {"HCPostNormF16": 3}, (stem, count)
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    report = dict(status="native_finite_bitwise_and_actual_dispatch_qualified",
                  fixture_pairs=len(manifest), repetitions=3,
                  finite_bitwise_elements=sum(f["elements_per_run"] for f in manifest) * 3,
                  actual_candidate_dispatches=len(manifest) * 3,
                  fixture_set=args.fixture_set,
                  provider_sha256=build["provider_sha256"],
                  quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
