import argparse
import collections
import json
from pathlib import Path
import re
import subprocess

from build_qsa_prefill_score16 import FLAG
from build_qmoe_grouped_prefill import FLAG as GROUPED_FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from profile_qualified_prefill import profile_command
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qsa_prefill_score16_native import PROGRAM
from run_qmoe_grouped_prefill_native import PROGRAMS as GROUPED_PROGRAMS
from build_qmoe_prefill_rows4 import PROGRAM as ROWS4_PROGRAM


def model_command(command, prefix, baseline_prefix, provider, mode, profile, grouped_prefill=False):
    result = profile_command(command, prefix, baseline_prefix)
    old_provider = "/workspace/qwen38-int4-dense-block32/build-v2/libonnxruntime_providers_webgpu.so"
    assert result[-1].count(old_provider) == 1
    exports = f"export {FLAG}=1; export {GROUPED_FLAG}={mode}; " if grouped_prefill else f"export {FLAG}={mode}; "
    result[-1] = exports + result[-1].replace(old_provider, str(provider).replace(
        "/home/sayanshaw/repos", "/workspace"))
    if not profile:
        marker = f" --profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile"
        assert result[-1].count(marker) == result[-1].count("--tokens 2") == 1
        result[-1] = result[-1].replace(marker, "").replace("--tokens 2", "--tokens 36")
    return result


def prefill_program_count(events):
    return sum(event.get("cat") == "Kernel" and event["name"].endswith("&" + PROGRAM) and
               re.search(r"inputs\[0\] = \{1,1024,640\}", event.get("args", {}).get("shapes", "")) is not None
               for event in events)


def grouped_program_count(events):
    counts = collections.Counter(e["name"].rsplit("&", 1)[-1]
                                 for e in events if e.get("cat") == "Kernel")
    count = counts[GROUPED_PROGRAMS[0]]
    assert count > 0 and all(counts[name] == count for name in GROUPED_PROGRAMS), counts
    return count


def rows4_program_count(events, grouped_count):
    count = 0
    projections = collections.Counter()
    for event in events:
        if event.get("cat") != "Kernel" or not event["name"].endswith("&" + ROWS4_PROGRAM):
            continue
        shapes = event["args"]["shapes"]
        input_shape = re.search(r"inputs\[0\] = \{(\d+),(\d+)\}", shapes)
        output_shape = re.search(r"outputs\[0\] = \{(\d+),(\d+)\}", shapes)
        assert input_shape and output_shape, shapes
        rows, width = map(int, input_shape.groups())
        output_rows, columns = map(int, output_shape.groups())
        assert rows == output_rows and rows in (5120, 10240), shapes
        assert (width, columns) in ((640, 1280), (160, 2560)), shapes
        projections[width, columns] += 1
        count += 1
    assert count == 2 * grouped_count, (count, grouped_count)
    assert projections == {(640, 1280): grouped_count, (160, 2560): grouped_count}, projections
    return count


def main(grouped_prefill=False, prefill_rows4=False):
    assert not prefill_rows4 or grouped_prefill
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("profile", "screen"), default="profile")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--trial", type=int, choices=range(8), default=0)
    parser.add_argument("--attempt", type=int, choices=range(8), default=0)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    attempt_suffix = f"-attempt{args.attempt}" if args.attempt else ""
    if args.phase == "profile" and (not args.full_validation or args.mode != 1 or args.trial):
        parser.error("Active-path profiling requires full validation, selector 1 and trial 0")
    variant = "qmoe-grouped-prefill-v1" if grouped_prefill else "qsa-prefill-score16-v1"
    flag = GROUPED_FLAG if grouped_prefill else FLAG
    family = "qmoe-grouped-prefill" if grouped_prefill else "qsa-prefill-score16"
    provider = Path(f"/home/sayanshaw/repos/qwen38-int4-{family}/build-v1/libonnxruntime_providers_webgpu.so")
    if prefill_rows4:
        variant = "qmoe-prefill-rows4-v2"
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                        "libonnxruntime_providers_webgpu.so")
    native_path = ROOT / f"{variant}-native-gpu7-20261031.qualification.json"
    native = json.loads(native_path.read_text())
    parent_name = ("qsa-prefill-score16-v1-contexts-20261031" if grouped_prefill else
                   "dense-block32-v2-contexts-20261030")
    if prefill_rows4:
        parent_name = "qmoe-grouped-prefill-v1-gpu7-full-paired-screen2048-20261031-attempt1"
    parent = json.loads((ROOT / f"{parent_name}.qualification.json").read_text())
    if grouped_prefill:
        assert native["status"] == (
            "grouped_rows4_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified" if prefill_rows4 else
            "grouped_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified")
        assert native["original_layers"] == 6 and native["finite_bitwise_comparisons"] == 300810240
        assert native["rows_tested"] == [16, 128, 256, 512, 1023, 1024, 1025]
        assert native["invalid_flag_explicitly_rejected"] and native["pending_windows_tested"] == [1, 64]
        assert len(native["additional_program_dispatches"]) == 6
        expected_dispatches = {name: 16 for name in GROUPED_PROGRAMS}
        if prefill_rows4:
            expected_dispatches[ROWS4_PROGRAM] = 32
        assert all(counts == expected_dispatches
                   for counts in native["additional_program_dispatches"].values())
    else:
        assert native["status"] == "prefill_score16_exact_reference_outputs_and_admission_qualified"
        assert native["fixtures"] == 76 and native["reference_executions"] == 2736
        assert native["invalid_selector_rejected"] and native["original_shader_and_decode_admission_preserved"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(provider) == native["provider_sha256"]
    if prefill_rows4:
        build = json.loads((provider.parent / "build-manifest.json").read_text())
        assert build["baseline_byte_identical"] and build["provider_sha256"] == native["provider_sha256"]
        assert build["baseline_provider_sha256"] == parent["provider_sha256"]
        assert sha(provider.parent / "native_compare") == native["comparator_sha256"] == build["comparator_sha256"]
        for name, digest in build["source_sha256"].items():
            assert sha(provider.parent / name) == digest
        for name, digest in native["source_sha256"].items():
            assert sha(ROOT / name) == digest
        if args.context != 2048:
            for validation in ("full-", ""):
                screen = json.loads((ROOT / (
                    f"{variant}-gpu7-{validation}paired-screen2048-20261031{attempt_suffix}.qualification.json"
                )).read_text())
                assert screen["provider_sha256"] == native["provider_sha256"]
                assert screen["full_validation"] == bool(validation)
                assert screen["status"] == "paired_ttft_win_qualified"
                assert screen["ttft_win_screen_qualified"] and screen["all_pairs_favor_candidate"]
                assert screen["all_original_sequences_match"] and screen["zero_major_faults"]
    programs_per_chunk = 48 if grouped_prefill else 12
    active_field = "active_grouped_prefill_programs" if grouped_prefill else "active_prefill_score16_programs"
    if args.phase != "profile" or args.context != 2048:
        proof = json.loads((ROOT / (
            f"{variant}-mode1-gpu7-full-profile2048-20261031{attempt_suffix}.qualification.json")).read_text())
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["full_validation"] and proof["zero_major_faults"] and proof["original_output_prefixes_match"]
        assert proof["selector"] == flag and proof["selector_value"] == 1
        assert proof[active_field] == 4 * programs_per_chunk
        if prefill_rows4:
            assert proof["active_rows4_prefill_projections"] == 8 * programs_per_chunk
    suffix = f"-trial{args.trial}" if args.phase == "screen" else ""
    prefix = ROOT / (f"{variant}-mode{args.mode}-gpu7-"
                     f"{'full-' if args.full_validation else ''}{args.phase}{args.context}-20261031{suffix}{attempt_suffix}")
    if any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".qualification.json")):
        raise FileExistsError(prefix)
    if not args.full_validation:
        proof = json.loads((ROOT / (
            f"{variant}-mode{args.mode}-gpu7-full-screen{args.context}-20261031-trial{args.trial}{attempt_suffix}.qualification.json"
        )).read_text())
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["full_validation"] and proof["zero_major_faults"] and proof["original_output_prefixes_match"]
        assert proof["selector"] == flag and proof["selector_value"] == args.mode
    baseline = ROOT / f"dense-block32-v2-chunk1024-gpu7-full-standard{args.context}-20261030.guard.json"
    command = model_command(json.loads(baseline.read_text())["command"], prefix,
                            baseline.name.removesuffix(".guard.json"), provider, args.mode, args.phase == "profile",
                            grouped_prefill=grouped_prefill)
    if not args.full_validation:
        marker = " --webgpu-validation-mode full"
        assert command[-1].count(marker) == 1
        command[-1] = command[-1].replace(marker, "")
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
        "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json"))
    verify_metadata(raw, guard, native, parent, args.full_validation, chunk=1024)
    assert guard["command"] == command and raw["load_experiment_flags"][flag] == str(args.mode)
    if grouped_prefill:
        assert raw["load_experiment_flags"][FLAG] == "1"
    for required_flag in ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32"):
        assert raw["load_experiment_flags"][required_flag] == "1"
    assert sha(provider) == native["provider_sha256"]
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    references = [s for s in json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
                  if s["context"] == args.context]
    verify_tokens(raw, references, args.context, 2 if args.phase == "profile" else 36)
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
        selector=flag, selector_value=args.mode, trial=args.trial, attempt=args.attempt, full_validation=args.full_validation,
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        original_output_prefixes_match=True, zero_major_faults=True, cache_preflight=cache,
        native_proof_sha256=sha(native_path),
        instrumented_not_a_speed_claim=args.phase == "profile", quality_qualified=False, headline_eligible=False)
    sources = [prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]
    if args.phase == "profile":
        assert raw["profiling_scope"] == "session"
        profiles = list(ROOT.glob(prefix.name + "-profile_*.json"))
        assert len(profiles) == 1, profiles
        events = json.loads(profiles[0].read_text())
        count = grouped_program_count(events) if grouped_prefill else prefill_program_count(events)
        assert count == 2 * (args.context // 1024) * programs_per_chunk, count
        report[active_field] = count
        report["programs"] = dict(collections.Counter(e["name"].rsplit("&", 1)[-1]
                                                     for e in events if e.get("cat") == "Kernel"))
        if prefill_rows4:
            report["active_rows4_prefill_projections"] = rows4_program_count(events, count)
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
