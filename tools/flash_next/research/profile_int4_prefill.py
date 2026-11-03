import argparse
import collections
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-m-version", choices=("v3", "v4"))
    args = parser.parse_args()
    baseline = "expert-bitcast-unpack-v2-gpu7-standard2048-20261029"
    original = json.loads((ROOT / (baseline + ".guard.json")).read_text())["command"]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    proof = None
    if args.small_m_version:
        proof = json.loads((ROOT / f"small-m-prefill-{args.small_m_version}-native-tail-gpu7-20261029.qualification.json").read_text())
        provider = Path(f"/home/sayanshaw/repos/qwen38-int4-small-m-prefill/build-{args.small_m_version}/libonnxruntime_providers_webgpu.so")
        assert sha(provider) == proof["provider_sha256"]
    for chunk in ((16,) if proof else (16, 64)):
        variant = f"small-m-prefill-{args.small_m_version}" if proof else "prefill"
        prefix = ROOT / f"{variant}-chunk{chunk}-gpu7-profile128-20261029"
        if any(prefix.with_suffix(s).exists() for s in (".guard.json", ".json", ".diagnostic.json")):
            raise FileExistsError(prefix)
        command = original.copy()
        assert "--contexts 2048" in command[-1] and "--tokens 256 --repetitions 3" in command[-1]
        command[-1] = command[-1].replace(baseline, prefix.name).replace(
            "--contexts 2048", "--contexts 128").replace(
            "--tokens 256 --repetitions 3",
            f"--tokens 2 --repetitions 1 --chunk {chunk} --webgpu-validation-mode full "
            f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")
        if proof:
            old_provider = "/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/libonnxruntime_providers_webgpu.so"
            assert command[-1].count(old_provider) == 1
            command[-1] = "export ORT_WEBGPU_SMALL_M_PREFILL=1; " + command[-1].replace(
                old_provider, str(provider).replace("/home/sayanshaw/repos", "/workspace"))
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
            "--output-prefix", str(prefix), "--", *command], check=True)
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        assert guard["status"] == raw["status"] == "completed" and guard["exit_code"] == 0
        assert guard["command"] == command and raw["chunk"] == chunk
        profiles = list(ROOT.glob(prefix.name + "-profile_*.json"))
        assert len(profiles) == 1, profiles
        dispatched = None
        if proof:
            assert raw["plugin_sha256"] == proof["provider_sha256"]
            dispatched = collections.Counter(event["name"] for event in json.loads(profiles[0].read_text())
                                            if event.get("cat") == "Kernel")
            assert any("SmallMPrefillMatMulNBitsF16" in name for name in dispatched), dispatched
            assert any("SmallMPrefillMatMulNBitsF32" in name for name in dispatched), dispatched
            assert all(v["major_page_faults"] == 0 for sample in raw["samples"]
                       for v in sample["process_resource_usage"].values())
        report = dict(status="prefill_profile_diagnostic", chunk=chunk, context=128,
            profile=str(profiles[0]), cache_preflight=cache,
            small_m_version=args.small_m_version, dispatched_programs=dispatched,
            process_faults=[s["process_resource_usage"] for s in raw["samples"]],
            no_frozen_reference_for_short_context=True, instrumented_timing_only=True,
            quality_qualified=False, headline_eligible=False)
        prefix.with_suffix(".diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "chunk", "profile")}), flush=True)


if __name__ == "__main__":
    main()
