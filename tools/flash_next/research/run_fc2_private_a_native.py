import json
from pathlib import Path
import shlex
import subprocess

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fc2-private-a/build-v1")
FLAG = "ORT_WEBGPU_QMOE_FC2_PRIVATE_A"


if __name__ == "__main__":
    output = ROOT / "fc2-private-a-v1-native-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((BUILD / "build-manifest.json").read_text())
    for path, digest in manifest["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed build source: {path}")
    if sha(BUILD / "libonnxruntime_providers_webgpu.so") != manifest["provider_sha256"]:
        raise RuntimeError("Changed candidate provider")
    fixture_proof = ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json"
    parent = json.loads(fixture_proof.read_text())
    fixtures = [Path("/home/sayanshaw/repos") / p for p in parent["fixture_sha256"]]
    if len(fixtures) != 23 or any(sha(p) != parent["fixture_sha256"][str(p.relative_to("/home/sayanshaw/repos"))] for p in fixtures):
        raise RuntimeError("Changed or incomplete native fixtures")
    uuid = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError("Invalid GPU0 UUID")
    comparator = host_path(manifest["native_comparator"])
    evidence = [Path(__file__), BUILD / "build-manifest.json", fixture_proof, comparator, GUARD]
    phases = []
    for flag in (0, 1, 2):
        prefix = ROOT / f"fc2-private-a-v1-native-flag{flag}-gpu0"
        dump = prefix.with_name(prefix.name + "-shaders.txt")
        for path in (prefix.with_suffix(".log"), prefix.with_suffix(".guard.json"), dump):
            if path.exists():
                raise FileExistsError(path)
        selected = fixtures if flag != 2 else fixtures[:1]
        arguments = ["/workspace/" + str(p.relative_to("/home/sayanshaw/repos"))
                     for p in [comparator, BUILD / "libonnxruntime_providers_webgpu.so", *selected]]
        shell = (
            "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
            + shlex.quote(uuid) + "; export "
            f"{FLAG}={flag} "
            "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
            "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
            "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1 "
            "ORT_WEBGPU_QSA_PARALLEL_MAP=1 ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1 "
            "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
            "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0 "
            f"ORT_WEBGPU_EP_SHADER_DUMP_FILE=/evidence/webgpu-perf-oct20/{dump.name}; exec "
            + shlex.join(arguments))
        command = ["taskset", "-c", "0-7", "bash", "-lc", shell]
        subprocess.run(["free", "-h"], check=True)
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
        child = subprocess.run(["python3", str(GUARD), "--gpu-index", "0", "--idle-seconds", "3",
            "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix), "--", *command])
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        log = prefix.with_suffix(".log").read_text()
        if guard["command"] != command or guard["gpu_uuid"] != uuid:
            raise RuntimeError("Wrong command/adapter")
        evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
        if flag == 2:
            if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in log:
                raise RuntimeError("Invalid private-A selector did not reject")
            phases.append(dict(flag=flag, invalid_selector_rejected=True))
            continue
        child.check_returncode()
        if (log.count("PASS changing_inputs_and_routes ") != 23 or
                "PASS total_exact_elements=9748480; CPU fallback disabled" not in log):
            raise RuntimeError("Incomplete finite bitwise/native routing coverage")
        shader = dump.read_text()
        marker = "=== WebGPU Shader code [QmoeFC2PrivateABitcastF16"
        if (flag == 0 and marker in shader) or (flag == 1 and marker not in shader):
            raise RuntimeError("Wrong private-A shader routing")
        if flag:
            begin = shader.index(marker)
            end = shader.index("] End ===", begin)
            selected_shader = shader[begin:end]
            if "cached_a" not in selected_shader or "var<workgroup> tile_a" in selected_shader:
                raise RuntimeError("Large shared activation staging was not removed")
        evidence.append(dump)
        phases.append(dict(flag=flag, finite_bitwise_elements=9748480, fixtures=23,
            cpu_fallback_disabled=True, private_a_shader_built=bool(flag),
            gpu_peak_gib=guard["peak_gpu_used_bytes"] / 1024**3))
    result = dict(status="fc2_private_a_native_finite_bitwise_and_shader_routing_qualified",
        phases=phases, provider_sha256=manifest["provider_sha256"],
        reference="original unfused GPU FP16 arithmetic, not independent whole-model CPU oracle",
        model_eligible=False, performance_qualified=False, quality_qualified=False, default_enabled=False,
        source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], phases=phases)), flush=True)
