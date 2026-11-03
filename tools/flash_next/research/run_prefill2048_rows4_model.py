import argparse
import collections
import json
from pathlib import Path
import re
import subprocess

from build_qmoe_prefill2048_rows4 import FLAG, PROGRAM, ROWS
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_qsa_score16_standard import verify_standard
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import gate_rows, verify_metadata, verify_prefill_flags, verify_tokens
from run_qmoe_grouped_prefill_native import PROGRAMS
from run_qmoe_prefill_rows4_standard import DIRECTORY as CONTROL_DIRECTORY
from run_qmoe_prefill_rows4_standard import FOLD_FLAG, GROUP_FLAG, TEMPLATE, standard_command


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill2048-rows4/build-v2")
VARIANT = "prefill2048-rows4-v2"
STAMP = "20261031"


def model_command(template, provider, prefix, context, full, control=False, profile=False):
    command = standard_command(template, provider, prefix, context, full)
    assert command[-1].count("--chunk 1024") == 1
    if not control:
        command[-1] = command[-1].replace("--chunk 1024", "--chunk 2048")
    assert FLAG not in command[-1]
    command[-1] = f"export {FLAG}={int(not control)} ORT_WEBGPU_QMOE_PREFILL2048=0; " + command[-1]
    if profile:
        assert full and not control
        assert command[-1].count("--tokens 256 --repetitions 3") == 1
        command[-1] = command[-1].replace("--tokens 256 --repetitions 3", "--tokens 2 --repetitions 1")
        command[-1] += f" --profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile"
    return command


def verify_flags(raw, control=False):
    flags = raw["load_experiment_flags"]
    assert flags[FLAG] == str(int(not control))
    assert flags["ORT_WEBGPU_QMOE_PREFILL2048"] == "0"
    verify_prefill_flags(raw, 1024 if control else 2048, selector=None if control else FLAG)
    for name in (GROUP_FLAG, "ORT_WEBGPU_QSA_PREFILL_SCORE16", "ORT_WEBGPU_DSA_QUERY_CACHE",
                 "ORT_WEBGPU_DENSE_BLOCK32"):
        assert flags[name] == "1", name
    assert flags[FOLD_FLAG] == "0"


def verify_large_profile(events):
    counts = collections.Counter(e["name"].rsplit("&", 1)[-1] for e in events if e.get("cat") == "Kernel")
    expected = 2 * 48
    assert all(counts[name] == expected for name in PROGRAMS), counts
    assert counts[PROGRAM] == 2 * expected and counts["MatMulNBitsGroupedRows4F16"] == 0, counts
    large_gates = [e for e in events if e.get("cat") == "Kernel"
                   and e["name"].endswith("&QmoeGate1Token") and gate_rows(e) == 2048]
    assert len(large_gates) == expected, len(large_gates)
    projections = collections.Counter()
    for event in events:
        if event.get("cat") != "Kernel" or not event["name"].endswith("&" + PROGRAM):
            continue
        shapes = event["args"]["shapes"]
        inp = re.search(r"inputs\[0\] = \{(\d+),(\d+)\}", shapes)
        out = re.search(r"outputs\[0\] = \{(\d+),(\d+)\}", shapes)
        assert inp and out, shapes
        rows, width = map(int, inp.groups())
        output_rows, columns = map(int, out.groups())
        assert rows == output_rows == 20480, shapes
        projections[width, columns] += 1
    assert projections == {(640, 1280): expected, (160, 2560): expected}, projections
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("profile", "standard"), default="profile")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.phase == "profile" and (not args.full_validation or args.control or args.context != 2048):
        parser.error("Path profiling requires the full-validation 2K candidate")
    mode = "full-" if args.full_validation else ""
    case = "control1024" if args.control else "candidate2048"
    prefix = ROOT / f"{VARIANT}-{case}-gpu7-{mode}{args.phase}{args.context}-trial{args.trial}-{STAMP}"
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / f"qmoe-{VARIANT}-native-gpu7-{STAMP}.qualification.json"
    attention_path = ROOT / f"{VARIANT}-dsa-native-gpu7-fixturesv1-{STAMP}.qualification.json"
    parent, native, attention = (json.loads(path.read_text()) for path in
                                 (parent_path, native_path, attention_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["status"] == "prefill2048_rows4_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    assert native["finite_bitwise_comparisons"] == 678297600 and native["rows_tested"] == list(ROWS)
    assert native["original_layers"] == 6 and native["pending_windows_tested"] == [1, 64]
    assert native["invalid_flag_explicitly_rejected"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert attention["status"] == "prefill2048_dsa_cpu_reference_and_admission_qualified"
    assert attention["provider_sha256"] == native["provider_sha256"]
    assert attention["expected_rejections"] == 12 and attention["invalid_selector_rejected"]
    assert attention["original_dsa_object_byte_identical"] and attention["shader_arithmetic_unchanged"]
    sources = [parent_path, native_path, attention_path, Path(__file__)]
    for proof in (native, attention):
        for name, digest in proof["source_sha256"].items():
            path = ROOT / name
            assert sha(path) == digest, path
    for version in (1, 2):
        directory = DIRECTORY.parent / f"build-v{version}"
        manifest_path = directory / "build-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        sources.append(manifest_path)
        for name, digest in manifest["source_sha256"].items():
            assert sha(directory / name) == digest, name
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        path = ROOT / f"{VARIANT}-context{previous}-screen-{STAMP}.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["status"] == "bracketed_ttft_gain_and_decode_preservation_both_modes"
        assert proof["provider_sha256"] == native["provider_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
        sources.append(path)
    if args.phase != "profile" and not args.control:
        path = ROOT / f"{VARIANT}-candidate2048-gpu7-full-profile2048-trial0-{STAMP}.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["programs"][PROGRAM] == 192
        sources.append(path)
        if not args.full_validation:
            path = ROOT / f"{VARIANT}-{case}-gpu7-full-standard{args.context}-trial{args.trial}-{STAMP}.qualification.json"
            proof = json.loads(path.read_text())
            assert proof["status"] == "original_sequences_qualified" and proof["zero_major_faults"]
            assert proof["provider_sha256"] == native["provider_sha256"]
            sources.append(path)
    provider = (CONTROL_DIRECTORY if args.control else DIRECTORY) / "libonnxruntime_providers_webgpu.so"
    digest = parent["provider_sha256"] if args.control else native["provider_sha256"]
    assert sha(provider) == digest
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    sources.append(template_path)
    command = model_command(json.loads(template_path.read_text())["command"], provider, prefix,
                            args.context, args.full_validation, args.control, args.phase == "profile")
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
    assert guard["command"] == command
    selected_native = {"provider_sha256": digest}
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    sources.append(controls_path)
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    chunk = 1024 if args.control else 2048
    verify_flags(raw, args.control)
    if args.phase == "profile":
        verify_metadata(raw, guard, selected_native, parent, True, chunk=chunk)
        assert raw["profiling"] and raw["profiling_scope"] == "session"
        result = verify_tokens(raw, references, args.context, 2)
        profiles = list(ROOT.glob(prefix.name + "-profile_*.json"))
        assert len(profiles) == 1, profiles
        programs = verify_large_profile(json.loads(profiles[0].read_text()))
        sources += profiles
    else:
        result = verify_standard(raw, guard, selected_native, parent, references,
                                 args.context, args.full_validation, chunk=chunk)
        programs = None
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources += [prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
                  control=args.control, full_validation=args.full_validation, trial=args.trial, chunk=chunk,
                  provider_sha256=digest, graph_sha256=parent["graph_sha256"],
                  config_sha256=parent["config_sha256"], result=result, programs=programs,
                  original_output_prefixes_match=True, zero_major_faults=True,
                  comparison_kind="separate_load_observation_not_isolated_route_gain",
                  instrumented_not_a_speed_claim=args.phase == "profile",
                  cache_preflight=cache, quality_qualified=False, headline_eligible=False,
                  source_sha256={path.name if path.parent == ROOT else str(path): sha(path)
                                 for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("status", "context", "phase", "control", "full_validation", "result")}), flush=True)


if __name__ == "__main__":
    main()
