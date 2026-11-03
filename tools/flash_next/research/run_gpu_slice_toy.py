import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-gpu-greedy-argmax/build-v6-r1")
PROVIDER = Path("/home/sayanshaw/repos/qwen38-int4-gpu-slice-copy/build-v1")
FLAG = "ORT_GENAI_WEBGPU_GPU_SLICE_COPY"


if __name__ == "__main__":
    output = ROOT / "gpu-slice-v6-r1-toy-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    manifests = [json.loads((p / "build-manifest.json").read_text()) for p in (BUILD, PROVIDER)]
    for manifest in manifests:
        for path, digest in manifest["source_sha256"].items():
            if sha(host_path(path)) != digest:
                raise RuntimeError(f"Changed input: {path}")
    if sha(BUILD / "libonnxruntime-genai.so") != manifests[0]["runtime_sha256"]:
        raise RuntimeError("Changed runtime")
    if sha(PROVIDER / "libonnxruntime_providers_webgpu.so") != manifests[1]["provider_sha256"]:
        raise RuntimeError("Changed provider")
    reference = ROOT / "gpu-greedy-argmax-v5-toy-probe4-gpu4"
    original = json.loads(reference.with_suffix(".guard.json").read_text())
    phases, baseline_checks, evidence = [], None, [Path(__file__), ROOT / "gpu_slice_toy.py", GUARD,
        BUILD / "build-manifest.json", PROVIDER / "build-manifest.json"]
    for flag in (0, 1, 2):
        prefix = ROOT / f"gpu-slice-v6-r1-toy-flag{flag}-gpu4"
        for suffix in (".log", ".json", ".guard.json"):
            if prefix.with_suffix(suffix).exists():
                raise FileExistsError(prefix.with_suffix(suffix))
        command = original["command"].copy()
        shell = command[-1].replace(reference.name, prefix.name)
        shell = replace_once(shell, "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=1;",
            f"ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=0; export {FLAG}={flag} ORT_WEBGPU_SLICE_UINT32_BITCOPY=1;")
        shell = shell.replace("qwen38-int4-gpu-greedy-argmax/build-v5", "qwen38-int4-gpu-greedy-argmax/build-v6-r1")
        shell = replace_once(shell, "gpu_greedy_toy_v5_v3.py", "gpu_slice_toy.py")
        shell = replace_once(shell,
            "/workspace/qwen38-int4-gpu-greedy-argmax/build-v6-r1/libonnxruntime_providers_webgpu.so",
            "/workspace/qwen38-int4-gpu-slice-copy/build-v1/libonnxruntime_providers_webgpu.so")
        first = "/workspace/qwen38-int4-gpu-greedy-argmax/toy-v5-v2/float32-32"
        if flag != 2:
            models = [f"/workspace/qwen38-int4-gpu-greedy-argmax/toy-v5-v2/{dtype}-{n}"
                      for dtype in ("float16", "float32") for n in (32, 2049, 248320)]
            shell = replace_once(shell, first, " ".join(models))
        command[-1] = shell
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
                raise RuntimeError("Invalid slice selector did not explicitly reject")
            phases.append(dict(flag=flag, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        result = json.loads(prefix.with_suffix(".json").read_text())
        if result["runtime_sha256"] != manifests[0]["runtime_sha256"] or result["gpu_slice_flag"] != str(flag):
            raise RuntimeError("Wrong runtime or selector")
        if len(result["checks"]) != 108 or result["score_elements_checked"] != 14022456:
            raise RuntimeError("Incomplete ID or initial/next score coverage")
        if flag == 0:
            baseline_checks = result["checks"]
        elif result["checks"] != baseline_checks:
            raise RuntimeError("Sliced copy changed greedy/API/penalty/seeded sampling behavior")
        calls = log.count("WEBGPU_GPU_SLICE_COPY ")
        if (flag == 0 and calls) or (flag == 1 and calls != 160):
            raise RuntimeError(f"Unexpected GPU slice coverage: {calls}")
        if "WEBGPU_GREEDY_MIN_LENGTH_FUSED" in log:
            raise RuntimeError("Unqualified fused helper stacked")
        evidence.append(prefix.with_suffix(".json"))
        for path, digest in result["source_and_evidence_sha256"].items():
            if sha(host_path(path)) != digest:
                raise RuntimeError(f"Changed fixture: {path}")
            evidence.append(host_path(path))
        phases.append(dict(flag=flag, ids_checked=len(result["checks"]),
            score_elements_checked=result["score_elements_checked"], gpu_slice_calls=calls,
            nan_payload_differences=result["nan_payload_differences"],
            gpu_peak_gib=guard["peak_gpu_used_bytes"] / 1024**3))
    result = dict(status="aligned_gpu_slice_native_ids_scores_penalties_sampling_exact", phases=phases,
        runtime_sha256=manifests[0]["runtime_sha256"], provider_sha256=manifests[1]["provider_sha256"],
        default_enabled=False, model_eligible=False, performance_qualified=False, quality_qualified=False,
        f16_nan_payload_cpu_identity_qualified=False,
        source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], phases=phases)), flush=True)
