import collections
import json
from pathlib import Path
import shlex
import subprocess

from prepare_generic_matmul_replay import SHAPES
from qualify_quickgelu_mul_long_context import ROOT, sha


def main():
    repos = Path("/home/sayanshaw/repos")
    directory = repos / "qwen38-int4-prefill-dense-stack/build-v1"
    build = json.loads((directory / "build-manifest.json").read_text())
    assert sha(directory / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]
    evidence = {}
    for kind in ("dense", "prefill"):
        prefix = ROOT / f"prefill-dense-stack-v1-{kind}-native-gpu7-20261029"
        if prefix.with_suffix(".guard.json").exists():
            raise FileExistsError(prefix)
        if kind == "dense":
            component = repos / "qwen38-int4-dense-bitcast-unpack/build-v1"
            previous = json.loads((ROOT / "dense-bitcast-unpack-v1-native-gpu7-20261029.qualification.json").read_text())
            fixtures = sorted((repos / "qwen38-int4-generic-matmul-replay/fixtures-v1").glob("*.onnx"))
            assert len(fixtures) == 27
            assert all(sha(p) == previous["fixture_sha256"][p.name] for p in fixtures)
            expected = sum(n for _, n in SHAPES) * 19 * 128
        else:
            component = repos / "qwen38-int4-qmoe-extended-prefill/build-v1"
            previous = json.loads((ROOT / "qmoe-extended-prefill-v1-native-gpu7-20261029.qualification.json").read_text())
            fixtures = sorted((repos / "qwen38-int4-qmoe-extended-prefill/fixtures-v1").glob("*.onnx"))
            assert len(fixtures) == 6
            assert all(sha(p) == previous["fixture_manifest"]["fixtures"][p.name]["fixture_sha256"] for p in fixtures)
            expected = 33454080
        assert previous["provider_sha256"] == build["component_provider_sha256"][kind]
        assert sha(component / "native_compare") == previous["comparator_sha256"]
        stats = ROOT / (prefix.name + "-stats.jsonl")
        command = (
            "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
            "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
            "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
            "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
            "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
            "ORT_WEBGPU_QMOE_INT4_PREFILL=1 ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 "
            "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1 "
            "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1 ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL=1 "
            "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
            "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 "
            f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +
            shlex.join(str(p).replace(str(repos), "/workspace") for p in [
                component / "native_compare", directory / "libonnxruntime_providers_webgpu.so", *fixtures]))
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
            "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        log = prefix.with_suffix(".log").read_text()
        assert guard["status"] == "completed" and guard["exit_code"] == 0
        assert "FAIL" not in log and "GetCapability" not in log
        assert f"PASS total_exact_elements={expected}; CPU fallback disabled" in log
        if kind == "dense":
            assert log.count("changing_rounds=128 invalid_flag_rejected=1") == 27
            rows = [json.loads(line) for line in stats.read_text().splitlines()]
            assert len(rows) == len(SHAPES)
            assert all(r["operator"] == "MatMulNBitsExact" and r["requests"] == 256 and
                       r["entries"] == 2 and r["capacity"] == 64 and
                       r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
                       r["hits"] + r["captures"] == r["requests"] for r in rows), rows
        else:
            assert log.count("retained_chunk16_reference=1 invalid_flag_rejected=1") == 6
            profiles = [Path(line.removeprefix("PROFILE ").replace("/evidence/webgpu-perf-oct20", str(ROOT)))
                        for line in log.splitlines() if line.startswith("PROFILE ")]
            assert len(profiles) == 6
            for path in profiles:
                counts = collections.Counter(e["name"].split("&")[-1] for e in json.loads(path.read_text())
                                             if e.get("cat") == "Kernel")
                assert counts["QmoeGate1Token"] == 152, (path, counts)
        evidence[kind] = dict(finite_bitwise_comparisons=expected, fixture_sha256={p.name: sha(p) for p in fixtures},
                              comparator_sha256=sha(component / "native_compare"))
    report = dict(status="combined_dense_and_prefill_native_qualified", evidence=evidence,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        quality_qualified=False, headline_eligible=False)
    output = ROOT / "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
