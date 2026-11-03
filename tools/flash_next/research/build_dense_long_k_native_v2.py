import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-dense-tile-parallel3-long-k/native-full-v2")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = ROOT / "dense_long_k_fixed_native_v2.cpp"
    binary = OUTPUT / "native_compare"
    command = [
        "c++", "-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror", str(source),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-lonnxruntime", "-o", str(binary),
    ]
    subprocess.run(command, check=True)
    with (OUTPUT / "generation.json").open("x") as stream:
        json.dump({"status": "built_not_qualified", "compile_command": command,
                   "binary_sha256": sha(binary),
                   "source_sha256": {str(path): sha(path) for path in (Path(__file__), source, ROOT / "dense_long_k_fixed_native.cpp",
                                                                  ROOT / "build_dense_long_k_native.py",
                                                                  ROOT / "prepare_dense_long_k_native_v2.py")}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
