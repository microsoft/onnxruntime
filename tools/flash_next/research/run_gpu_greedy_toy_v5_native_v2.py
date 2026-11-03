import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-gpu-greedy-argmax/build-v5")
FLAG = "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"


if __name__ == "__main__":
    output = ROOT / "gpu-greedy-argmax-v5-toy-native-v2-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, digest in manifest["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed build source: {path}")
    for filename, key in (("libonnxruntime-genai.so", "runtime_sha256"),
                          ("libonnxruntime_providers_webgpu.so", "provider_sha256")):
        if sha(BUILD / filename) != manifest[key]:
            raise RuntimeError("Changed native-test binary")
    original_prefix = ROOT / "gpu-greedy-argmax-v5-toy-probe4-gpu4"
    guard = json.loads(original_prefix.with_suffix(".guard.json").read_text())
    probe = json.loads(original_prefix.with_suffix(".json").read_text())
    if guard["status"] != "completed" or probe["status"] != "toy_runtime_ids_scores_cpu_authority_exact":
        raise RuntimeError("Tiny-model routing/API probe prerequisite missing")
    if probe["runtime_sha256"] != manifest["runtime_sha256"]:
        raise RuntimeError("Wrong probe runtime")
    evidence = [Path(__file__), manifest_path, GUARD, original_prefix.with_suffix(".json"),
                original_prefix.with_suffix(".guard.json"), ROOT / "gpu_greedy_toy_v5_v4.py",
                BUILD / "libonnxruntime-genai.so", BUILD / "libonnxruntime_providers_webgpu.so"]
    phases = []
    for flag in (0, 1, 2):
        prefix = ROOT / f"gpu-greedy-argmax-v5-toy-native-v2-flag{flag}-gpu4"
        for suffix in (".json", ".log", ".guard.json"):
            if prefix.with_suffix(suffix).exists():
                raise FileExistsError(prefix.with_suffix(suffix))
        command = [part.replace(original_prefix.name, prefix.name).replace(
            "gpu_greedy_toy_v5_v3.py", "gpu_greedy_toy_v5_v4.py") for part in guard["command"]]
        if command[-1].count(f"{FLAG}=1;") != 1:
            raise RuntimeError("Nonunique fused-selector replacement")
        command[-1] = command[-1].replace(f"{FLAG}=1;", f"{FLAG}={flag};")
        first = "/workspace/qwen38-int4-gpu-greedy-argmax/toy-v5-v2/float32-32"
        models = [f"/workspace/qwen38-int4-gpu-greedy-argmax/toy-v5-v2/{dtype}-{n}"
                  for dtype in ("float16", "float32") for n in (32, 2049, 248320)]
        if command[-1].count(first) != 1:
            raise RuntimeError("Nonunique native model-list replacement")
        if flag != 2:
            command[-1] = command[-1].replace(first, " ".join(models))
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
        child = subprocess.run(["python3", str(GUARD), "--gpu-index", "4", "--idle-seconds", "3",
            "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix), "--", *command])
        state = json.loads(prefix.with_suffix(".guard.json").read_text())
        log = prefix.with_suffix(".log").read_text()
        if state["command"] != command or state["gpu_index"] != 4:
            raise RuntimeError("Wrong native command/adapter")
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"Invalid value for environment variable {FLAG}: 2." not in log:
                raise RuntimeError("Fused selector2 did not explicitly reject")
            phases.append(dict(flag=flag, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        result = json.loads(prefix.with_suffix(".json").read_text())
        if (result["status"] != "toy_runtime_ids_non_nan_score_bits_nan_positions_cpu_authority_exact" or
                result["runtime_sha256"] != manifest["runtime_sha256"] or result["fused_flag"] != str(flag)):
            raise RuntimeError("Wrong native result/runtime/selector")
        if len(result["checks"]) != 108 or result["score_elements_checked"] != 7011228:
            raise RuntimeError("Incomplete native ID/score shape coverage")
        fused, cached = log.count("WEBGPU_GREEDY_MIN_LENGTH_FUSED"), log.count("WEBGPU_GREEDY_ARGMAX_CACHED")
        if (flag == 0 and (fused or cached)) or (flag == 1 and (fused < 84 or cached != 90)):
            raise RuntimeError(f"Incomplete fused/cached routing: fused={fused} cached={cached}")
        evidence.append(prefix.with_suffix(".json"))
        for path, digest in result["source_and_evidence_sha256"].items():
            if sha(host_path(path)) != digest:
                raise RuntimeError(f"Changed toy source/model: {path}")
            evidence.append(host_path(path))
        phases.append(dict(flag=flag, ids_checked=len(result["checks"]),
                           api_score_elements_checked=result["score_elements_checked"],
                           fused_min_length_calls=fused, cached_gpu_selections=cached,
                           gpu_peak_gib=state["peak_gpu_used_bytes"] / 1024**3,
                           non_nan_score_bits_exact=result["non_nan_score_bits_exact"],
                           nan_positions_exact=result["nan_positions_exact"],
                           nan_payload_differences=result["nan_payload_differences"]))
    result = dict(status="coalesced_helper_toy_ids_api_numeric_scores_penalties_sampling_exact", phases=phases,
        runtime_sha256=manifest["runtime_sha256"], provider_sha256=manifest["provider_sha256"],
        f16_nan_payload_cpu_identity_qualified=False,
        model_eligible=False, performance_qualified=False, quality_qualified=False, default_enabled=False,
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], phases=phases)), flush=True)
