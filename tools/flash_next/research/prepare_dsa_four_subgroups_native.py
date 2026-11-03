import json
from pathlib import Path
import shutil
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
PROOF = Path("/workspace/qwen38-int4-dsa-four-subgroups/native-proof-v1")


if __name__ == "__main__":
    PROOF.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / "dsa_constant_original_compare.cpp", ROOT / "hc_original_reference_io.inc"]
    for source in sources:
        shutil.copy2(source, PROOF / source.name)
    binary = PROOF / "independent_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(PROOF / sources[0].name),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report = dict(status="constant_attention_original_comparator_built",
        comparator_sha256=sha(binary), constant_fixtures=True, changing_inputs=False,
        source_sha256={str(source): sha(source) for source in [Path(__file__), *sources]},
        copied_sha256={source.name: sha(PROOF / source.name) for source in sources})
    (PROOF / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], flush=True)
