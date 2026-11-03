import argparse
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_qmoe_prefill_rows4 import VARIANT
from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_qsa_score16_standard import verify_standard
from run_prefill256_model import verify_prefill_flags


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2")
TEMPLATE = "qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031"
OLD_PROVIDER = "/workspace/qwen38-int4-qsa-prefill-score16/build-v1/libonnxruntime_providers_webgpu.so"
GROUP_FLAG = "ORT_WEBGPU_QMOE_GROUPED_PREFILL"
FOLD_FLAG = "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD"


def standard_command(command, provider, prefix, context, full):
    assert context in (2048, 8192, 32768)
    result = command.copy()
    script = result[-1]
    for marker in (OLD_PROVIDER, TEMPLATE + ".json", TEMPLATE + "-stats.jsonl",
                   TEMPLATE + "-static-stats.jsonl", "--contexts 2048",
                   "--tokens 256 --repetitions 3", "--chunk 1024",
                   " --webgpu-validation-mode full"):
        assert script.count(marker) == 1, marker
    assert script.count(TEMPLATE) == 3
    assert GROUP_FLAG not in script and FOLD_FLAG not in script
    assert "--compare-" not in script and "--profile-" not in script
    script = script.replace(OLD_PROVIDER, str(provider).replace("/home/sayanshaw/repos", "/workspace"))
    script = script.replace(TEMPLATE, prefix.name).replace("--contexts 2048", f"--contexts {context}")
    if not full:
        script = script.replace(" --webgpu-validation-mode full", "")
    result[-1] = f"export {GROUP_FLAG}=1 {FOLD_FLAG}=0; " + script
    return result


def verify_result(raw, guard, native, parent, references, context, full):
    result = verify_standard(raw, guard, native, parent, references, context, full, chunk=1024)
    verify_prefill_flags(raw, 1024)
    flags = raw["load_experiment_flags"]
    for flag in (GROUP_FLAG, "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32",
                 "ORT_WEBGPU_QSA_PREFILL_SCORE16"):
        assert flags[flag] == "1", flag
    assert flags[FOLD_FLAG] == "0"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    prefix = ROOT / (f"{VARIANT}-chunk1024-gpu7-{'full-' if args.full_validation else ''}"
                     f"standard{args.context}-20261031")
    for suffix in (".json", ".guard.json", ".qualification.json", ".log"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    parent_path = ROOT / f"{VARIANT}-contexts-20261031.qualification.json"
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    assert parent["status"] == "controlled_ttft_screens_all_contexts_qualified_not_standard_decode"
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert set(parent["contexts"]) == {"2048", "8192", "32768"}
    assert native["finite_bitwise_comparisons"] == 300810240
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"] == native["provider_sha256"]
    for path, digest in parent["source_sha256"].items():
        assert sha(Path(path)) == digest, path
    prerequisites = []
    if not args.full_validation:
        prerequisites.append(ROOT / f"{VARIANT}-chunk1024-gpu7-full-standard{args.context}-20261031.qualification.json")
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        prerequisites.extend(ROOT / f"{VARIANT}-chunk1024-gpu7-{mode}standard{previous}-20261031.qualification.json"
                             for mode in ("full-", ""))
    for path in prerequisites:
        proof = json.loads(path.read_text())
        assert proof["status"] == "original_sequences_qualified"
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = standard_command(json.loads(template_path.read_text())["command"],
                               provider, prefix, args.context, args.full_validation)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    result = verify_result(raw, guard, native, parent, references, args.context, args.full_validation)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    report = dict(status="original_sequences_qualified", context=args.context, phase="standard",
                  full_validation=args.full_validation, chunk=1024, zero_major_faults=True,
                  original_output_prefixes_match=True, provider_sha256=native["provider_sha256"],
                  graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
                  result=result, fixed_grouped_prefill_selector=1, rejected_fc2_fold_selector=0,
                  native_proof_sha256=sha(native_path), parent_proof_sha256=sha(parent_path),
                  command_template_sha256=sha(template_path), controls_sha256=sha(controls_path),
                  source_sha256={path.name: sha(path) for path in
                                 (prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"),
                                  Path(__file__))},
                  cache_preflight=cache, output_tokens=256, measured_repetitions=3,
                  excluded_warmup_requests=1, original_sequence_comparisons=1024,
                  comparison_kind="standalone_standard_not_isolated_causal_pair",
                  quality_qualified=False, headline_eligible=False)
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "context", "full_validation", "result")}), flush=True)


if __name__ == "__main__":
    main()
