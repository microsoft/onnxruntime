import argparse
import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT
from qualify_row4_cpu_locality import compare_samples
from run_gpu_greedy_argmax_ready_standard import BUILD, FLAG, VARIANT, prefix_for, verify_proof, verify_workers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    output = ROOT / f"{VARIANT}-context{args.context}-{'repeat-' if args.repeat else ''}screen.qualification.json"
    execution = output.with_name(output.name.replace(".qualification.json", "-execution.json"))
    if output.exists() or execution.exists():
        raise FileExistsError(output)
    build = json.loads((BUILD / "build-manifest.json").read_text())
    sources, rows, records = {}, [], []
    if args.repeat:
        initial_path = ROOT / f"{VARIANT}-context{args.context}-screen.qualification.json"
        initial = verify_proof(initial_path, "initial_gpu_greedy_decode_gain_both_modes")
        assert initial["runtime_sha256"] == build["runtime_sha256"]
        sources[str(initial_path)] = sha(initial_path)
    manifest = dict(status="running", context=args.context, repeat=args.repeat, records=records)
    execution.write_text(json.dumps(manifest, indent=2) + "\n")
    baseline_trial = int(args.repeat)
    provider_hash = None
    for full in (True, False):
        samples, results, times = [], [], []
        for mode, trial in (("control", baseline_trial), ("gpu", baseline_trial),
                            ("control", baseline_trial + 1)):
            prefix = prefix_for(mode, args.context, full, trial)
            if not (args.repeat and mode == "control" and trial == baseline_trial):
                command = ["python3", str(ROOT / "run_gpu_greedy_argmax_ready_standard.py"),
                           "--mode", mode, "--context", str(args.context), "--trial", str(trial)]
                if full:
                    command.append("--full-validation")
                record = dict(command=command, prefix=str(prefix), completed=False)
                records.append(record)
                execution.write_text(json.dumps(manifest, indent=2) + "\n")
                subprocess.run(command, check=True)
                record["completed"] = True
                execution.write_text(json.dumps(manifest, indent=2) + "\n")
            proof_path = prefix.with_suffix(".qualification.json")
            proof = verify_proof(proof_path, "gpu_greedy_original_prefixes_qualified")
            raw, guard = [json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json")]
            assert proof["mode"] == mode and proof["trial"] == trial
            assert proof["context"] == args.context and proof["full_validation"] == full
            assert proof["same_runtime_and_provider_both_sides"] and proof["control_is_lanes8_with_gpu_greedy_disabled"]
            assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
            assert verify_workers(raw, "control") == proof["workers"]
            assert proof["experiment_flag"] == FLAG and proof["experiment_flag_value"] == int(mode == "gpu")
            assert guard["status"] == "completed" and guard["exit_code"] == 0
            assert proof["runtime_sha256"] == raw["genai_native"]["sha256"] == build["runtime_sha256"]
            provider_hash = proof["provider_sha256"] if provider_hash is None else provider_hash
            assert proof["provider_sha256"] == raw["plugin_sha256"] == provider_hash
            assert raw["load_experiment_flags"][FLAG] == str(int(mode == "gpu"))
            sources.update(proof["source_sha256"])
            sources[str(proof_path)] = sha(proof_path)
            measured = [sample for sample in raw["samples"] if not sample["warmup"]]
            assert len(raw["samples"]) == 4 and len(measured) == 3
            for sample in measured:
                assert sample["output_tokens"] == 256
                assert abs(sample["decode_tps"] * sample["decode_seconds"] - 255) < 1e-8
            samples.append(measured)
            results.append(proof["result"])
            times.append((proof["started_ns"], proof["completed_ns"]))
        assert times[0][1] <= times[1][0] and times[1][1] <= times[2][0]
        rows.append(dict(full_validation=full, before=results[0], candidate=results[1],
                         after=results[2], comparison=compare_samples(samples[1], samples[0], samples[2])))
    supported = all(row["comparison"]["supported"] for row in rows)
    manifest["status"] = "completed"
    execution.write_text(json.dumps(manifest, indent=2) + "\n")
    report = dict(status=(("repeated_" if args.repeat else "initial_") + "gpu_greedy_decode_gain_both_modes"
                          if supported else "no_supported_decode_gain"),
        context=args.context, repeat=args.repeat, provider_sha256=provider_hash,
        runtime_sha256=build["runtime_sha256"], rows=rows,
        same_runtime_and_provider_both_sides=True, control_is_lanes8_with_gpu_greedy_disabled=True,
        original_output_prefixes_match=True, zero_major_faults=True,
        all_candidate_modes_above_100_decode_tps=all(row["candidate"]["median_decode_tps"] > 100 for row in rows),
        comparison_kind="bracketed_separate_load_same_runtime_selector_screen",
        fresh_reuses_initial_ending_controls=args.repeat, quality_qualified=False, headline_eligible=False,
        source_sha256={**sources, **{str(path): sha(path) for path in (Path(__file__), execution)}})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "repeat", "rows")}), flush=True)


if __name__ == "__main__":
    main()
