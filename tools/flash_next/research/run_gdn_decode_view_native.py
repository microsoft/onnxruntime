import argparse
import collections
import json
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    directory = Path("/home/sayanshaw/repos/qwen38-int4-gdn-decode-view")
    prefix = ROOT / "gdn-decode-view-native-20261029"
    if prefix.with_suffix(".guard.json").exists() and not args.validate_existing:
        raise FileExistsError(prefix)
    fixtures = json.loads((directory / "fixtures.json").read_text())
    build = json.loads((directory / "build-original/build-manifest.json").read_text())
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 "
        "ORT_TEST_TOGGLE_GDN_VIEW=1 "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile; "
        "exec /workspace/qwen38-int4-gdn-decode-view/build-original/native_compare "
        "/workspace/qwen38-int4-gdn-decode-view/build-original/libonnxruntime_providers_webgpu.so " +
        " ".join("/workspace/qwen38-int4-gdn-decode-view/fixtures/" + f["fixture"] for f in fixtures))
    if not args.validate_existing:
        subprocess.run(
            ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
             "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "240",
             "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count("PASS exact finite outputs ") == len(fixtures)
    for fixture in fixtures:
        stem = Path(fixture["fixture"]).stem
        for mode in ("reference", "fused"):
            profiles = list(ROOT.glob(prefix.name + "-profile-" + stem + "-" + mode + "*.json"))
            assert len(profiles) == 1, profiles
            counts = collections.Counter(e["name"].split("&")[-1]
                                         for e in json.loads(profiles[0].read_text())
                                         if e.get("cat") == "Kernel")
            assert counts["GatedDeltaNet"] == 3, (stem, counts)
            expected_unpack = 0 if mode == "fused" and fixture["decode_view"] else 3
            assert counts["GatedDeltaNetUnpackQkv"] == expected_unpack, (stem, mode, counts)
    assert sha(directory / "build-original/libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    report = dict(status="native_finite_bitwise_view_and_fallback_qualified",
                  fixtures=len(fixtures), view_cases=sum(f["decode_view"] for f in fixtures),
                  finite_bitwise_elements=sum(f["elements_per_run"] for f in fixtures) * 3,
                  eliminated_unpack_dispatches=sum(f["decode_view"] for f in fixtures) * 3,
                  provider_sha256=build["provider_sha256"], quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
