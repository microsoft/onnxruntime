import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from run_gpu_greedy_argmax_initial_model_v4 import ROOT, command_for
from prepare_webgpu_int4_metrics import GUARD


if __name__ == "__main__":
    prefix = ROOT / "gpu-greedy-argmax-v5-toy-probe1-gpu4"
    for suffix in (".json", ".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    command = command_for(prefix, 1, 1)
    benchmark = "/evidence/webgpu-perf-oct20/benchmark_gpu_greedy_argmax.py"
    if command[-1].count(benchmark) != 1:
        raise RuntimeError("Nonunique toy benchmark replacement")
    command[-1] = command[-1].split(benchmark)[0] + (
        "/evidence/webgpu-perf-oct20/gpu_greedy_toy_v5.py "
        "--models /workspace/qwen38-int4-gpu-greedy-argmax/toy-v5/float32-32 "
        "--plugin /workspace/qwen38-int4-gpu-greedy-argmax/build-v5/libonnxruntime_providers_webgpu.so "
        f"--output /evidence/webgpu-perf-oct20/{prefix.name}.json")
    command = [part.replace("48-55", "56-63").replace(
        "/workspace/qwen38-int4-gpu-greedy-argmax/build-v4",
        "/workspace/qwen38-int4-gpu-greedy-argmax/build-v5") for part in command]
    command[-1] = "export ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=1; " + command[-1]
    source = ROOT / "gpu_greedy_toy_v5.py"
    prefix.with_name(prefix.name + "-sources.json").write_text(
        json.dumps({str(p): sha(p) for p in (Path(__file__), source)}, indent=2) + "\n")
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    subprocess.run(["python3", str(GUARD), "--gpu-index", "4", "--idle-seconds", "3",
                    "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix),
                    "--", *command], check=True)
