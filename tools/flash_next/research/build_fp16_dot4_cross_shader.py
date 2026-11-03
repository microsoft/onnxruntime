import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-cross-shader/build-v1")
PROOF = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


if __name__ == "__main__":
    proof = json.loads((PROOF / "baseline.qualification.json").read_text())
    for name, digest in proof["sdk_sha256"].items():
        if sha(PROOF / name) != digest:
            raise RuntimeError(f"Changed SDK input: {name}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = OUTPUT / "cross_shader_compare.cpp"
    source.write_text((ROOT / "fp16_dot4_cross_shader_compare.cpp").read_text())
    binary = OUTPUT / "cross_shader_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(source),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report = dict(status="built_not_qualified", sdk_provenance_checked=True,
                  source_sha256=sha(source), comparator_sha256=sha(binary),
                  provider_changed=False, model_eligible=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
