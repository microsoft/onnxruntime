import argparse
import json
from pathlib import Path
import subprocess

from build_dense_output_shape import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_dense_output_shape_model import model_command
from run_qmoe_prefill_rows4_standard import TEMPLATE, verify_result


VARIANT = "dense-output-shape-v2-net-original"


def prefix_for(mode, context, full, trial, phase="standard"):
    assert context == 2048 and phase == "standard"
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}standard2048-trial{trial}-20261031"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "shape"), required=True)
    parser.add_argument("--context", type=int, choices=(2048,), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.mode == "shape" and args.trial != 0:
        parser.error("Only the initial original-provider net screen is supported")
    prefix = prefix_for(args.mode, 2048, args.full_validation, args.trial)
    for suffix in (".json", ".guard.json", ".qualification.json", ".log"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    inherited_path = ROOT / "dense-output-shape-v2-context2048-repeat-screen-20261031.qualification.json"
    inherited = json.loads(inherited_path.read_text())
    assert inherited["status"] == "repeated_shape_decode_gain_both_modes"
    for name, digest in inherited["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    if args.mode == "control":
        native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
        directory = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2")
    else:
        native_path = ROOT / "dense-output-shape-v2-native-gpu7-20261031.qualification.json"
        directory = Path("/home/sayanshaw/repos/qwen38-int4-dense-output-shape/build-v2")
    native = json.loads(native_path.read_text())
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == native["provider_sha256"]
    expected = parent["provider_sha256"] if args.mode == "control" else inherited["provider_sha256"]
    assert native["provider_sha256"] == expected
    build_path = directory / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["provider_sha256"] == expected
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest, name
    prerequisites = [inherited_path]
    if not args.full_validation:
        path = prefix_for(args.mode, 2048, True, args.trial).with_suffix(".qualification.json")
        full = json.loads(path.read_text())
        assert full["status"] == "original_sequences_qualified" and full["provider_sha256"] == expected
        for name, digest in full["source_sha256"].items():
            assert sha(Path(name)) == digest, name
        prerequisites.append(path)
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = model_command(json.loads(template_path.read_text())["command"], provider, prefix,
                            2048, args.full_validation, args.mode, "standard")
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json"))
    assert guard["command"] == command and raw["comparison"] is None
    assert raw["cpu_affinity"] == list(range(96)) and "worker_affinity_override" not in raw
    assert any(len(mask) == 1 for mask in raw["thread_cpu_affinity"].values())
    assert raw["genai_native"]["sha256"] == sha(Path(
        "/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/build-h200-int4-legacy/libonnxruntime-genai.so"))
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [s for s in json.loads(controls_path.read_text())["samples"] if s["context"] == 2048]
    result = verify_result(raw, guard, native, parent, references, 2048, args.full_validation)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources = [Path(__file__), ROOT / "run_dense_output_shape_model.py", ROOT / "benchmark_webgpu_int4.py",
               parent_path, native_path, build_path, template_path, controls_path, *prerequisites,
               *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    report = dict(status="original_sequences_qualified", context=2048, mode=args.mode,
                  phase="standard", trial=args.trial, full_validation=args.full_validation,
                  provider_sha256=expected, baseline_provider_sha256=parent["provider_sha256"],
                  original_provider_control=args.mode == "control",
                  graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
                  experiment_flag=FLAG, experiment_flag_value=int(args.mode == "shape"),
                  original_output_prefixes_match=True, zero_major_faults=True, result=result,
                  worker_policy_unchanged=True, cache_preflight=cache,
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "mode", "result")}), flush=True)


if __name__ == "__main__":
    main()
