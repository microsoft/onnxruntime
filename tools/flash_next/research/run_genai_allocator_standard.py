import argparse
import json
from pathlib import Path
import subprocess
import time

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_genai_allocator_workers import FLAG, OLD_RUNTIME, RUNTIME, verify_workers
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command, verify_result


VARIANT = "genai-allocator-single-thread-v1"
STAMP = "20261031"


def prefix_for(mode, context, full, trial):
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}standard{context}-trial{trial}-{STAMP}"


def allocator_command(template, provider, prefix, context, full, mode):
    assert mode in ("control", "single")
    command = standard_command(template, provider, prefix, context, full)
    assert command[-1].count(OLD_RUNTIME) == 1
    command[-1] = f"export {FLAG}={int(mode == 'single')}; " + command[-1].replace(
        OLD_RUNTIME, str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace"))
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "single"), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.mode == "single" and args.trial == 2:
        parser.error("Trial2 is reserved for the final control")
    prefix = prefix_for(args.mode, args.context, args.full_validation, args.trial)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    build_path = RUNTIME / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["baseline_relink_byte_identical"] and build["baseline_rebuild_byte_identical"]
    runtime = RUNTIME / "libonnxruntime-genai.so"
    assert sha(runtime) == build["runtime_sha256"]
    for name, digest in build["source_sha256"].items():
        path = Path(name.replace("/workspace", "/home/sayanshaw/repos", 1).replace(
            "/evidence", str(ROOT.parent), 1))
        assert sha(path) == digest, name
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"] == native["provider_sha256"]
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    prerequisites = [ROOT / f"{VARIANT}-{mode}-workers-gpu7-{STAMP}.qualification.json"
                     for mode in ("control", "single")]
    if not args.full_validation:
        prerequisites.append(prefix_for(args.mode, args.context, True, args.trial).with_suffix(".qualification.json"))
    if args.mode == "single" and args.trial == 1:
        prerequisites.append(ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json")
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        prerequisites.append(ROOT / f"{VARIANT}-context{previous}-repeat-screen-{STAMP}.qualification.json")
    for path in prerequisites:
        proof = json.loads(path.read_text())
        expected = ("allocator_worker_policy_original_prefixes_qualified" if "-workers-" in path.name else
                    "repeated_allocator_decode_gain_both_modes" if "repeat-screen" in path.name else
                    "initial_allocator_decode_gain_both_modes" if "-screen-" in path.name else
                    "allocator_standard_original_prefixes_qualified")
        assert proof["status"] == expected and proof["provider_sha256"] == native["provider_sha256"]
        assert proof["runtime_sha256"] == build["runtime_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = allocator_command(json.loads(template_path.read_text())["command"], provider, prefix,
                                args.context, args.full_validation, args.mode)
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
    raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    assert guard["command"] == command and raw["genai_native"]["sha256"] == build["runtime_sha256"]
    assert raw["comparison"] is None and not raw["profiling"] and raw["profiling_scope"] is None
    workers = verify_workers(raw, args.mode)
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    result = verify_result(raw, guard, native, parent, references, args.context, args.full_validation)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources = [Path(__file__), ROOT / "run_genai_allocator_workers.py", ROOT / "benchmark_webgpu_int4.py",
               parent_path, native_path, build_path, runtime, template_path, controls_path, *prerequisites,
               *[prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log")]]
    report = dict(status="allocator_standard_original_prefixes_qualified", mode=args.mode,
                  context=args.context, trial=args.trial, full_validation=args.full_validation,
                  runtime_sha256=build["runtime_sha256"], provider_sha256=native["provider_sha256"],
                  experiment_flag=FLAG, experiment_flag_value=int(args.mode == "single"),
                  workers=workers, original_output_prefixes_match=True, zero_major_faults=True,
                  main_affinity_unchanged=True, manual_affinity_override=False,
                  model_and_arithmetic_unchanged=True, quality_qualified=False, headline_eligible=False,
                  result=result, started_ns=started_ns, completed_ns=completed_ns,
                  cache_preflight=cache, source_sha256={str(path): sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "mode", "context", "full_validation", "result", "workers")}), flush=True)


if __name__ == "__main__":
    main()
