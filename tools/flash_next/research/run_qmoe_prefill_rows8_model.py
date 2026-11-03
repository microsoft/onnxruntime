import argparse
import collections
import json
from pathlib import Path
import re
import subprocess

from build_qmoe_prefill_rows8 import FLAG, PROGRAM
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_prefill_score16_model import grouped_program_count, model_command


VARIANT = "qmoe-prefill-rows8-v1"
PARENT = "qmoe-prefill-rows4-v2"
GROUP_FLAG = "ORT_WEBGPU_QMOE_GROUPED_PREFILL"
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows8/build-v1")


def verify_profile(events, context):
    grouped = grouped_program_count(events)
    assert grouped == 96 * (context // 1024)
    projections = collections.Counter()
    for event in events:
        if event.get("cat") != "Kernel":
            continue
        name = event["name"].rsplit("&", 1)[-1]
        assert name != "MatMulNBitsGroupedRows4F16"
        if name != PROGRAM:
            continue
        shapes = event["args"]["shapes"]
        input_shape = re.search(r"inputs\[0\] = \{(\d+),(\d+)\}", shapes)
        output_shape = re.search(r"outputs\[0\] = \{(\d+),(\d+)\}", shapes)
        assert input_shape and output_shape, shapes
        rows, width = map(int, input_shape.groups())
        output_rows, columns = map(int, output_shape.groups())
        assert rows == output_rows and rows in (5120, 10240), shapes
        projections[width, columns] += 1
    assert projections == {(640, 1280): grouped, (160, 2560): grouped}, projections
    return dict(active_grouped_prefill_programs=grouped, active_rows8_prefill_projections=2 * grouped)


def command_for_case(template, baseline, prefix, provider, mode, profile, full):
    command = model_command(template, prefix, baseline, provider, 1, profile, grouped_prefill=True)
    command[-1] = f"export {FLAG}={mode} ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD=0; " + command[-1]
    if not full:
        marker = " --webgpu-validation-mode full"
        assert command[-1].count(marker) == 1
        command[-1] = command[-1].replace(marker, "")
    return command


def verify_win_prerequisite(proof, provider, context, full):
    assert proof["status"] == "paired_ttft_win_qualified"
    assert proof["provider_sha256"] == provider
    assert proof["context"] == context and proof["full_validation"] == full
    assert proof["selector"] == FLAG and proof["fixed_grouped_prefill_selector"] == 1
    assert proof["ttft_win_screen_qualified"] and proof["all_pairs_favor_candidate"]
    assert proof["all_original_sequences_match"] and proof["zero_major_faults"]
    for name, digest in proof["source_sha256"].items():
        assert sha(ROOT / name) == digest, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("profile", "screen"), default="profile")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--trial", type=int, choices=range(4), default=0)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if args.phase == "profile" and (not args.full_validation or args.mode != 1 or args.trial):
        parser.error("Profiling requires full validation, selector 1 and trial 0")
    prefix = ROOT / (f"{VARIANT}-mode{args.mode}-gpu7-{'full-' if args.full_validation else ''}"
                     f"{args.phase}{args.context}-20261031" +
                     (f"-trial{args.trial}" if args.phase == "screen" else ""))
    for suffix in (".json", ".guard.json", ".qualification.json", ".log"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    parent_path = ROOT / f"{PARENT}-contexts-20261031.qualification.json"
    native, parent = (json.loads(p.read_text()) for p in (native_path, parent_path))
    assert native["status"] == "grouped_rows8_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    assert native["original_layers"] == 6 and native["finite_bitwise_comparisons"] == 300810240
    assert native["invalid_flag_explicitly_rejected"] and native["pending_windows_tested"] == [1, 64]
    assert native["selector"] == FLAG
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["baseline_byte_identical"] and build["replay_variant_keyed"]
    assert sha(provider) == build["provider_sha256"] == native["provider_sha256"]
    assert sha(DIRECTORY / "native_compare") == build["comparator_sha256"] == native["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest
    for name, digest in native["source_sha256"].items():
        assert sha(ROOT / name) == digest
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        for context in (2048, previous):
            for full in (True, False):
                path = ROOT / f"{VARIANT}-gpu7-{'full-' if full else ''}paired-screen{context}-20261031.qualification.json"
                verify_win_prerequisite(json.loads(path.read_text()), native["provider_sha256"], context, full)
    if args.phase != "profile" or args.context != 2048:
        proof = json.loads((ROOT / f"{VARIANT}-mode1-gpu7-full-profile2048-20261031.qualification.json").read_text())
        assert proof["provider_sha256"] == native["provider_sha256"] and proof["zero_major_faults"]
        assert proof["active_rows8_prefill_projections"] == 384 and proof["full_validation"]
    if not args.full_validation:
        proof = json.loads((ROOT / f"{VARIANT}-mode{args.mode}-gpu7-full-screen{args.context}-20261031"
                            f"-trial{args.trial}.qualification.json").read_text())
        assert proof["provider_sha256"] == native["provider_sha256"] and proof["zero_major_faults"]
        assert proof["original_output_prefixes_match"] and proof["selector_value"] == args.mode
    baseline = ROOT / f"dense-block32-v2-chunk1024-gpu7-full-standard{args.context}-20261030.guard.json"
    command = command_for_case(json.loads(baseline.read_text())["command"],
                               baseline.name.removesuffix(".guard.json"), prefix, provider, args.mode,
                               args.phase == "profile", args.full_validation)
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
    verify_metadata(raw, guard, native, parent, args.full_validation, chunk=1024)
    assert guard["command"] == command
    assert raw["load_experiment_flags"][FLAG] == str(args.mode)
    for required in (GROUP_FLAG, "ORT_WEBGPU_QSA_PREFILL_SCORE16", "ORT_WEBGPU_QMOE_PREFILL1024",
                     "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32"):
        assert raw["load_experiment_flags"][required] == "1"
    references = [s for s in json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
                  if s["context"] == args.context]
    verify_tokens(raw, references, args.context, 2 if args.phase == "profile" else 36)
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
                  selector=FLAG, selector_value=args.mode, fixed_grouped_prefill_selector=1, trial=args.trial,
                  full_validation=args.full_validation, provider_sha256=native["provider_sha256"],
                  baseline_provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
                  config_sha256=parent["config_sha256"], original_output_prefixes_match=True,
                  zero_major_faults=True, cache_preflight=cache, native_proof_sha256=sha(native_path),
                  build_manifest_sha256=sha(build_path), parent_proof_sha256=sha(parent_path),
                  instrumented_not_a_speed_claim=args.phase == "profile", quality_qualified=False,
                  headline_eligible=False)
    sources = [prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]
    if args.phase == "profile":
        assert raw["profiling_scope"] == "session"
        profiles = list(ROOT.glob(prefix.name + "-profile_*.json"))
        assert len(profiles) == 1
        report.update(verify_profile(json.loads(profiles[0].read_text()), args.context))
        sources += profiles
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    report["source_sha256"] = {p.name: sha(p) for p in (*sources, baseline, Path(__file__))}
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "selector_value", "zero_major_faults")}),
          flush=True)


if __name__ == "__main__":
    main()
