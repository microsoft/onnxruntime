from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once


root = Path("/workspace/qwen38-int4-hc-fused-sum")
output = root / "diagnostic-v1"
output.mkdir(exist_ok=False)
source = output / "native_compare.cpp"
original = (root / "build-v1/native_compare.cpp").read_text()
source.write_text(replace_once(original,
    'std::to_string(round) + " element=" + std::to_string(i)',
    '''std::to_string(round) + " output=" + std::to_string(output) +
    " element=" + std::to_string(i) +
    " expected_bits=" + std::to_string(half ? reinterpret_cast<const uint16_t*>(expected)[i] :
                                             reinterpret_cast<const uint32_t*>(expected)[i]) +
    " actual_bits=" + std::to_string(half ? reinterpret_cast<const uint16_t*>(observed)[i] :
                                           reinterpret_cast<const uint32_t*>(observed)[i])'''))
subprocess.run(["c++", "-O3", "-std=c++20", str(source),
    "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
    "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
    "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
    "-lonnxruntime", "-o", str(output / "native_compare")], check=True)
