import json
from pathlib import Path
import re
import shlex
import subprocess

from build_native_matrix_limit_probe import FLAG
from build_qmoe_node_replay import sha
from run_dense_paired_matrix_batch8_probe import FLAGS, run_guard
from run_fp16_dot4_tree_probe import UUID, workspace


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-native-matrix-limit-probe/build-v1")
DATA = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-batch8/fixtures-v1")
BINARY = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-batch8/build-v1/paired_matrix_compare")


if __name__ == "__main__":
    outcome = ROOT / "native-matrix-limit-probe-v1-outcome.json"
    if outcome.exists():
        raise FileExistsError(outcome)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in manifest["source_sha256"].items():
        if sha(BUILD / relative) != digest:
            raise RuntimeError(f"Changed diagnostic provenance: {relative}")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError("Changed diagnostic provider")
    comparator_manifest = json.loads((BINARY.parent / "build-manifest.json").read_text())
    if sha(BINARY) != comparator_manifest["paired_comparator_sha256"]:
        raise RuntimeError("Changed comparator")
    model = DATA / "k640-n2560.onnx"
    reference = DATA / "k640-n2560-reference.f16"
    reference_sha = sha(reference)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                    "--format=csv,noheader"], check=True)
    prefix = ROOT / "native-matrix-limit-probe-v1-gpu7"
    args = [workspace(BINARY), "compare", workspace(provider), workspace(model),
            workspace(reference), "/evidence/webgpu-perf-oct20/" + prefix.name + "-profile"]
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
               f"export {FLAGS} {FLAG}=1; exec " + shlex.join(args))
    code, log = run_guard(command, prefix)
    if code != 0 or "DENSE comparisons=163840 mismatches=0 K=640 N=2560" not in log:
        raise RuntimeError(f"Arithmetic-unchanged diagnostic failed: {prefix.with_suffix('.log')}")
    if sha(reference) != reference_sha:
        raise RuntimeError("Original reference changed")
    values = re.findall(r"NATIVE_MATRIX_DEVICE_LIMITS storage=(\d+) threads_x=(\d+) "
                        r"invocations=(\d+) pair_configuration=([01])", log)
    if len(values) != 64 or len(set(values)) != 1:
        raise RuntimeError(f"Missing or inconsistent actual device limits: {values}")
    storage, threads, invocations, configuration = map(int, values[0])
    evidence = [Path(__file__), ROOT / "build_native_matrix_limit_probe.py",
                manifest_path, provider, BINARY, BINARY.parent / "build-manifest.json",
                model, reference, prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")]
    result = dict(status="actual_device_limits_observed", gpu_uuid=UUID,
                  workgroup_storage_bytes=storage, workgroup_size_x=threads,
                  workgroup_invocations=invocations, paired_matrix_configuration=bool(configuration),
                  model_eligible=False, shader_arithmetic_unchanged=True,
                  default_kernel_comparisons=163840, mismatches=0,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
