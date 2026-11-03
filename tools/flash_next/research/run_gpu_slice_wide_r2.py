import json
import subprocess

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_gpu_slice_model import verify_sources


if __name__ == "__main__":
    native_path = ROOT / "gpu-slice-v6-r2-toy-outcome.json"
    native = verify_sources(native_path, "aligned_gpu_slice_native_ids_scores_penalties_sampling_exact")
    original = ROOT / "gpu-slice-v6-r2-toy-flag1-gpu4"
    prefix = ROOT / "gpu-slice-v6-r2-wide-gpu4"
    for suffix in (".json", ".log", ".guard.json", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    command = json.loads(original.with_suffix(".guard.json").read_text())["command"].copy()
    shell = command[-1].replace(original.name, prefix.name)
    shell = replace_once(shell, "gpu_slice_toy.py", "gpu_slice_wide.py")
    begin = shell.index(" --models ")
    end = shell.index(" --plugin ", begin)
    shell = shell[:begin] + shell[end:]
    command[-1] = shell
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    subprocess.run(["python3", str(GUARD), "--gpu-index", "4", "--idle-seconds", "3",
        "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix), "--", *command], check=True)
    raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
    if raw["runtime_sha256"] != native["runtime_sha256"] or guard["command"] != command:
        raise RuntimeError("Wrong wide-copy runtime/command")
    log = prefix.with_suffix(".log").read_text()
    expected = (f"WEBGPU_GPU_SLICE_COPY source_bytes={raw['source_bytes']} "
                f"offset_bytes={raw['source_bytes'] - raw['copied_bytes']} copied_bytes={raw['copied_bytes']}")
    if log.count(expected) != 4 or raw["source_start_word"] <= 2**31 or raw["source_word_count"] >= 2**32:
        raise RuntimeError("Missing large unsigned-index copy coverage")
    sources = [native_path, ROOT / "gpu_slice_wide.py", ROOT / "run_gpu_slice_wide_r2.py",
               GUARD, *[prefix.with_suffix(s) for s in (".json", ".log", ".guard.json")]]
    result = dict(status="segmented_large_unsigned_index_gpu_slice_exact", runtime_sha256=native["runtime_sha256"],
        provider_sha256=native["provider_sha256"], source_bytes=raw["source_bytes"],
        copied_bytes=raw["copied_bytes"], source_start_word=raw["source_start_word"],
        bitwise_score_elements_checked=raw["score_elements_checked"], gpu_slice_calls=4,
        gpu_peak_gib=guard["peak_gpu_used_bytes"] / 1024**3,
        model_eligible=False, quality_qualified=False, performance_qualified=False,
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_and_evidence_sha256"}), flush=True)
