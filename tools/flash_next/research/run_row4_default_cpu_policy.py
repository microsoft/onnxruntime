import argparse
import json
from pathlib import Path
import subprocess
import time

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command, verify_result
from run_row4_cpu_locality import CPU_MASKS, placement_command


VARIANT = "row4-default-cpu-policy-v1"
STAMP = "20261031"


def prefix_for(placement, context, full, trial):
    return ROOT / f"{VARIANT}-{placement}-gpu7-{'full-' if full else ''}standard{context}-trial{trial}-{STAMP}"


def cpu_command(template, provider, prefix, context, full, placement):
    command = standard_command(template, provider, prefix, context, full)
    if placement == "local8":
        command[-1] += " --inherit-worker-cpu-affinity"
    return placement_command(command, placement)


def verify_cpu_policy(raw, placement):
    expected = CPU_MASKS[placement]
    assert raw["cpu_affinity"] == expected
    masks = raw["thread_cpu_affinity"]
    assert masks and all(mask and set(mask) <= set(expected) for mask in masks.values())
    if placement == "baseline":
        assert "worker_affinity_override" not in raw
        assert any(len(mask) == 1 for mask in masks.values()), "Default individual worker pins are absent"
    else:
        assert raw["worker_affinity_override"]["target_cpus"] == expected
        assert all(mask == expected for mask in masks.values()), "Not all workers use the local8 mask"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", choices=("baseline", "local8"), default="baseline")
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.placement == "local8" and args.trial == 2:
        parser.error("Trial2 is reserved for the control after the fresh candidate")
    prefix = prefix_for(args.placement, args.context, args.full_validation, args.trial)
    for suffix in (".json", ".guard.json", ".qualification.json", ".log"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["finite_bitwise_comparisons"] == 300810240
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"] == native["provider_sha256"]
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    sources = [parent_path, native_path, Path(__file__), ROOT / "benchmark_webgpu_int4.py",
               ROOT / "cpu_local_windows.py", ROOT / "run_row4_cpu_locality.py"]
    prerequisites = []
    if args.placement == "local8" and args.trial == 1:
        path = ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["status"] == "initial_default_policy_decode_gain_both_modes"
        assert proof["provider_sha256"] == native["provider_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
        sources.append(path)
    if not args.full_validation:
        prerequisites.append(prefix_for(args.placement, args.context, True, args.trial).with_suffix(
            ".qualification.json"))
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        path = ROOT / f"{VARIANT}-context{previous}-repeat-screen-{STAMP}.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["status"] == "repeated_default_policy_decode_gain_both_modes"
        assert proof["provider_sha256"] == native["provider_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
        sources.append(path)
    for path in prerequisites:
        proof = json.loads(path.read_text())
        assert proof["status"] == "original_sequences_qualified" and proof["zero_major_faults"]
        assert proof["provider_sha256"] == native["provider_sha256"] and proof["placement"] == args.placement
        sources.append(path)
    topology = subprocess.run(["nvidia-smi", "topo", "-m"], text=True, capture_output=True, check=True).stdout
    gpu7 = next(line.split() for line in topology.splitlines() if line.startswith("GPU7"))
    assert gpu7[-3:] == ["48-95", "1", "N/A"], gpu7
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = cpu_command(json.loads(template_path.read_text())["command"], provider, prefix,
                          args.context, args.full_validation, args.placement)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    started_ns = time.time_ns()
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    completed_ns = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json"))
    assert guard["command"] == command
    verify_cpu_policy(raw, args.placement)
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    result = verify_result(raw, guard, native, parent, references, args.context, args.full_validation)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources += [template_path, controls_path, *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    report = dict(status="original_sequences_qualified", placement=args.placement, context=args.context,
                  full_validation=args.full_validation, trial=args.trial, result=result,
                  provider_sha256=native["provider_sha256"], graph_sha256=parent["graph_sha256"],
                  config_sha256=parent["config_sha256"], cpu_affinity=raw["cpu_affinity"],
                  thread_cpu_affinity=raw["thread_cpu_affinity"],
                  default_worker_policy_preserved=args.placement == "baseline",
                  started_ns=started_ns, completed_ns=completed_ns, gpu7_topology_row=gpu7,
                  process_only_affinity=True, memory_policy_unchanged=True,
                  original_output_prefixes_match=True, zero_major_faults=True, chunk=1024,
                  output_tokens=256, measured_repetitions=3, excluded_warmup_requests=1,
                  comparison_kind="default_policy_separate_load_cpu_placement_screen_not_kernel_gain",
                  cache_preflight=cache, quality_qualified=False, headline_eligible=False,
                  source_sha256={path.name: sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("status", "placement", "context", "full_validation", "result")}), flush=True)


if __name__ == "__main__":
    main()
