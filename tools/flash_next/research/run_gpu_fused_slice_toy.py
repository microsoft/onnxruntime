import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_private_a_native import BUILD as PROVIDER
from run_gpu_slice_model import verify_sources
from run_gpu_slice_toy_r2 import BUILD


FLAG = "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"


if __name__ == "__main__":
    output = ROOT / "gpu-fused-slice-v1-native-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    parent_path = ROOT / "gpu-slice-v6-r2-toy-outcome.json"
    parent = verify_sources(parent_path, "aligned_gpu_slice_native_ids_scores_penalties_sampling_exact")
    provider_path = ROOT / "fc2-private-a-v1-native-outcome.json"
    provider = verify_sources(provider_path, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    if (sha(BUILD / "libonnxruntime-genai.so") != parent["runtime_sha256"] or
            sha(PROVIDER / "libonnxruntime_providers_webgpu.so") != provider["provider_sha256"]):
        raise RuntimeError("Changed native-qualified binaries")
    template = ROOT / "gpu-slice-v6-r2-toy-flag1-gpu4"
    original = json.loads(template.with_suffix(".guard.json").read_text())["command"]
    reference = json.loads(template.with_suffix(".json").read_text())
    evidence = [Path(__file__), ROOT / "gpu_slice_toy.py", GUARD, parent_path, provider_path,
        BUILD / "libonnxruntime-genai.so", PROVIDER / "libonnxruntime_providers_webgpu.so"]
    phases = []
    for flag in (0, 1, 2):
        prefix = ROOT / f"gpu-fused-slice-v1-native-flag{flag}-gpu4"
        for suffix in (".log", ".json", ".guard.json"):
            if prefix.with_suffix(suffix).exists():
                raise FileExistsError(prefix.with_suffix(suffix))
        command = original.copy()
        command[-1] = command[-1].replace(template.name, prefix.name)
        command[-1] = replace_once(command[-1], f"{FLAG}=0;", f"{FLAG}={flag};")
        command[-1] = replace_once(command[-1],
            "/workspace/qwen38-int4-gpu-slice-copy/build-v1/libonnxruntime_providers_webgpu.so",
            "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so")
        command[-1] = "export ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0; " + command[-1]
        subprocess.run(["free", "-h"], check=True)
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
        child = subprocess.run(["python3", str(GUARD), "--gpu-index", "4", "--idle-seconds", "3",
            "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix), "--", *command])
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        log = prefix.with_suffix(".log").read_text()
        if guard["command"] != command or guard["gpu_index"] != 4:
            raise RuntimeError("Wrong command/adapter")
        evidence.extend(prefix.with_suffix(s) for s in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"Invalid value for environment variable {FLAG}: 2." not in log:
                raise RuntimeError("Invalid fused selector did not reject")
            phases.append(dict(flag=flag, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        if (raw["runtime_sha256"] != parent["runtime_sha256"] or raw["fused_flag"] != str(flag) or
                raw["gpu_slice_flag"] != "1" or raw["checks"] != reference["checks"] or
                len(raw["checks"]) != 108 or raw["score_elements_checked"] != 14022456 or
                log.count("WEBGPU_GPU_SLICE_COPY ") != 160):
            raise RuntimeError("Incomplete combined helper API/ID/slice coverage")
        fused = log.count("WEBGPU_GREEDY_MIN_LENGTH_FUSED")
        if (flag == 0 and fused) or (flag == 1 and not fused):
            raise RuntimeError("Wrong coalesced helper routing")
        evidence.append(prefix.with_suffix(".json"))
        for p, digest in raw["source_and_evidence_sha256"].items():
            resolved = host_path(p)
            if sha(resolved) != digest:
                raise RuntimeError(f"Changed fixture: {p}")
            evidence.append(resolved)
        phases.append(dict(flag=flag, behavior_checks=108, score_elements_checked=14022456,
            gpu_slice_calls=160, fused_helper_calls=fused, nan_payload_differences=raw["nan_payload_differences"],
            sampled_peak_gpu_gib=guard["peak_gpu_used_bytes"] / 1024**3))
    result = dict(status="combined_fused_slice_ids_scores_penalties_sampling_exact", phases=phases,
        runtime_sha256=parent["runtime_sha256"], provider_sha256=provider["provider_sha256"],
        default_enabled=False, model_eligible=False, performance_qualified=False, quality_qualified=False,
        f16_nan_payload_cpu_identity_qualified=False,
        source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_and_evidence_sha256"}), flush=True)
