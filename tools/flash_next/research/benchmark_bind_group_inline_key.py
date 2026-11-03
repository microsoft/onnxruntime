import argparse
import json
import os
from pathlib import Path

from benchmark_gpu_slice import main, run_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    keys = ("ORT_WEBGPU_BIND_GROUP_INLINE_KEY", "ORT_WEBGPU_QMOE_FC2_PRIVATE_A",
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH", "ORT_GENAI_WEBGPU_GPU_SLICE_COPY",
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "ORT_WEBGPU_DYNAMIC_UNIFORMS")
    flags = {key: os.environ.get(key) for key in keys}
    expected = dict(zip(keys[1:], ("0", "0", "1", "8192", "1")))
    if flags[keys[0]] not in ("0", "1") or any(flags[key] != value for key, value in expected.items()):
        raise ValueError("Wrong supported baseline or inline-key selector")
    with Path(str(args.output) + ".launcher.json").open("x") as stream:
        stream.write(json.dumps(dict(status="flags_captured_before_model_load", flags=flags), indent=2) + "\n")
    main()
    if any(os.environ.get(key) != value for key, value in flags.items()):
        raise RuntimeError("Captured context settings changed during the benchmark")
