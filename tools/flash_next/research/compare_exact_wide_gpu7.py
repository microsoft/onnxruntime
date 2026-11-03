import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from qualify_quickgelu_mul_long_context import ROOT, sha, snapshot


GPU_UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-exact-wide-gpu7/build")
PREFIX = ROOT / "exact-wide-gpu7-windows-2k-20261028"
NATIVE = ROOT / "exact-wide-gpu7-native-20261028.qualification.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--cache-warm-log", type=Path)
    args = parser.parse_args()
    build = (Path(str(BUILD.parent) + "-filtered") / "build") if args.filtered else BUILD
    prefix = ROOT / ("exact-wide-filtered-gpu7-windows-2k-20261028" if args.filtered else PREFIX.name)
    if args.output_prefix:
        prefix = args.output_prefix
    native_path = ROOT / ("exact-wide-filtered-gpu7-native-20261028.qualification.json"
                          if args.filtered else NATIVE.name)
    if prefix.with_suffix(".guard.json").exists():
        raise FileExistsError(prefix)
    deadline = time.monotonic() + 2000
    while not native_path.exists():
        guard_path = native_path.with_suffix("").with_suffix(".guard.json")
        if guard_path.exists():
            native_guard = json.loads(guard_path.read_text())
            if native_guard["status"] in ("failed", "blocked_gpu_busy", "run_timed_out", "yielded_to_training"):
                raise RuntimeError(f"Native prerequisite failed: {native_guard['status']}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Native prerequisite did not qualify")
        time.sleep(5)
    native = json.loads(native_path.read_text())
    assert native["status"] == "native_finite_bitwise_qualified_not_model_qualified"
    warm_log = args.cache_warm_log or ROOT / "gpu7-original-data-cache-warm-20261028.log"
    while True:
        warmed = [json.loads(line) for line in warm_log.read_text().splitlines() if line.startswith("{")]
        if len(warmed) == 2:
            for entry in warmed:
                assert entry["bytes_read"] == Path(entry["path"]).stat().st_size
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU data cache warming did not complete")
        time.sleep(5)
    assert sha(build / "libonnxruntime_providers_webgpu.so") == native["provider_sha256"]
    invocation = json.loads((ROOT / "quickgelu-mul-candidate-32k-20261027.guard.json").read_text())["command"].copy()
    invocation[-1] = invocation[-1].replace(
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd", GPU_UUID
    ).replace("quickgelu-mul-candidate-32k-20261027", prefix.name).replace(
        "--contexts 32768", "--contexts 2048"
    ).replace("--tokens 256 --repetitions 3", "--tokens 288 --repetitions 4 --compare-exact-wide-windows").replace(
        "/workspace/qwen38-int4-quickgelu-next/build/libonnxruntime_providers_webgpu.so",
        "/workspace/qwen38-int4-exact-wide-gpu7/build/libonnxruntime_providers_webgpu.so",
    )
    if args.filtered:
        invocation[-1] = invocation[-1].replace(
            "/workspace/qwen38-int4-exact-wide-gpu7/build/",
            "/workspace/qwen38-int4-exact-wide-gpu7-filtered/build/")
    initial = snapshot()
    subprocess.run(
        ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
         "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1000",
         "--output-prefix", str(prefix), "--", *invocation], check=True,
    )
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    data = json.loads(prefix.with_suffix(".json").read_text())
    assert guard["status"] == data["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == 7 and guard["gpu_uuid"] == GPU_UUID
    assert data["plugin_sha256"] == native["provider_sha256"]
    reference = next(s for s in json.loads(
        (ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
        if s["context"] == 2048 and not s["warmup"] and s["repetition"] == 0)
    assert len(data["samples"]) == 4 and len(data["window_samples"]) == 32
    for sample in data["samples"]:
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert sample["output_token_ids"][:256] == reference["output_token_ids"]
        assert len(sample["output_token_ids"]) == 288 and sample["same_prompt_output_matches"]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
    summary = data["window_summary"]["2048"]
    assert summary["matched_window_pairs"] == 16
    assert math.isfinite(summary["geometric_mean_matched_window_ratio"])
    report = dict(status="complementary_windows_original_prefix_qualified",
                  scope="Not the standard 256-output TPS table; same-provider flag comparison",
                  summary=summary, all_four_original_256_token_prefixes_exact=True,
                  all_four_288_token_sequences_equal=True, zero_major_faults=True,
                  provider_sha256=data["plugin_sha256"], graph_sha256=data["graph_sha256"],
                  read_only_cache_warming=warmed,
                  initial_resources=initial, final_resources=snapshot(),
                  quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
