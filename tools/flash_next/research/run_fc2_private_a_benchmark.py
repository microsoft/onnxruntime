import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_private_a_native import BUILD, FLAG


if __name__ == "__main__":
    output = ROOT / "fc2-private-a-v1-operator-screen.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / "fc2-private-a-v1-native-outcome.json"
    native = json.loads(native_path.read_text())
    if native["status"] != "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified":
        raise RuntimeError("Missing native qualification")
    for p, digest in native["source_and_evidence_sha256"].items():
        if sha(Path(p)) != digest:
            raise RuntimeError(f"Changed native evidence: {p}")
    fixture = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-node-replay/fixtures/expert_layer0_rows1.onnx")
    parent = json.loads((ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json").read_text())
    relative = str(fixture.relative_to("/home/sayanshaw/repos"))
    if sha(fixture) != parent["fixture_sha256"][relative]:
        raise RuntimeError("Changed benchmark fixture")
    phases, evidence, baseline_hash = [], [Path(__file__), native_path, GUARD,
        ROOT / "fc2_private_a_benchmark.cpp", BUILD / "native_benchmark", fixture], None
    for flag, phase, profiling in ((0, "before", False), (1, "candidate", False),
                                   (0, "after", False), (0, "profile-control", True), (1, "profile-candidate", True)):
        prefix = ROOT / f"fc2-private-a-v1-{phase}-gpu0"
        for suffix in (".json", ".log", ".guard.json"):
            if prefix.with_suffix(suffix).exists():
                raise FileExistsError(prefix.with_suffix(suffix))
        template = ROOT / f"fc2-private-a-v1-native-flag{flag}-gpu0.guard.json"
        command = json.loads(template.read_text())["command"].copy()
        shell = command[-1]
        begin = shell.index("; exec ")
        arguments = ["/workspace/" + str(p.relative_to("/home/sayanshaw/repos"))
                     for p in (BUILD / "native_benchmark", BUILD / "libonnxruntime_providers_webgpu.so", fixture)]
        arguments += ["/evidence/webgpu-perf-oct20/" + prefix.name + ".json",
                      "/evidence/webgpu-perf-oct20/" + prefix.name + "-ort-profile" if profiling else "-"]
        shell = shell[:begin] + "; export ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1 ORT_WEBGPU_EP_SHADER_DUMP_FILE=; exec " + shlex.join(arguments)
        command[-1] = shell
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
        subprocess.run(["python3", str(GUARD), "--gpu-index", "0", "--idle-seconds", "3",
            "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix), "--", *command], check=True)
        raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
        if (raw["status"] != "qmoe_operator_benchmark_completed" or guard["command"] != command or
                raw["profiling"] != profiling or raw["measured_runs"] != 128 or raw["warmups"] != 16 or
                raw["measured_major_faults"] != 0):
            raise RuntimeError("Invalid benchmark scope/command/faults")
        if baseline_hash is None:
            baseline_hash = raw["output_hash"]
        elif raw["output_hash"] != baseline_hash:
            raise RuntimeError("Benchmark output bytes changed")
        evidence.extend(prefix.with_suffix(s) for s in (".json", ".log", ".guard.json"))
        record = dict(flag=flag, phase=phase, profiling=profiling,
                      median_operator_wall_us=statistics.median(raw["run_seconds"]) * 1e6,
                      run_seconds=raw["run_seconds"])
        if profiling:
            profile = host_path(raw["profile_path"])
            events = json.loads(profile.read_text())
            name = "QmoeFC2PrivateABitcastF16" if flag else "QmoeFC2FinalMixBitcastF16"
            matched = [event for event in events if name in json.dumps(event)]
            record["fc2_profile_events"] = matched
            evidence.append(profile)
        phases.append(record)
    before, candidate, after = [p["run_seconds"] for p in phases[:3]]
    ratios = [math.sqrt(b * a) / c for b, c, a in zip(before, candidate, after)]
    result = dict(status="fc2_private_a_operator_screen_completed", phases=phases,
        matched_operator_wall_speed_ratio=statistics.geometric_mean(ratios),
        positive_operator_runs=sum(r > 1 for r in ratios), measured_operator_runs=128,
        provider_sha256=native["provider_sha256"], output_hash=baseline_hash,
        model_eligible=False, performance_qualified=False, quality_qualified=False,
        operator_ratios_are_not_model_decode_ratios=True,
        source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("phases", "source_and_evidence_sha256")}), flush=True)
