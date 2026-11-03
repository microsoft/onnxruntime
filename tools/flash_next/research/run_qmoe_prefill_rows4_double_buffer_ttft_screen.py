import argparse
import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha
from qualify_qsa_prefill_score16_screens import ttft_metrics
from run_qmoe_prefill_rows4_double_buffer_model import prefix_for, VARIANT
from run_qmoe_prefill_rows4_double_buffer_native import BUILD, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    args = parser.parse_args()
    outcome = ROOT / f"{VARIANT}-context{args.context}-ttft-screen.qualification.json"
    execution = ROOT / f"{VARIANT}-context{args.context}-ttft-execution.json"
    if outcome.exists() or execution.exists():
        raise FileExistsError("Existing controlled TTFT screen")
    build_path = BUILD / "build-manifest.json"
    build = json.loads(build_path.read_text())
    rows, sources, identity = [], [Path(__file__), build_path], None
    sequences = {True: set(), False: set()}
    for full in (True, False):
        trials = []
        for trial, mode in enumerate((0, 1, 1, 0)):
            prefix = prefix_for(args.context, full, "screen", mode, trial)
            execution.write_text(json.dumps(dict(status="running", rows=rows, full_validation=full,
                                                 trial=trial, selector=mode), indent=2) + "\n")
            subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                            "--format=csv,noheader"], check=True)
            command = ["python3", str(ROOT / "run_qmoe_prefill_rows4_double_buffer_model.py"),
                       "--context", str(args.context), "--phase", "screen",
                       "--mode", str(mode), "--trial", str(trial)]
            if full:
                command.append("--full-validation")
            subprocess.run(command, check=True)
            proof_path = prefix.with_suffix(".qualification.json")
            proof = json.loads(proof_path.read_text())
            raw = json.loads(prefix.with_suffix(".json").read_text())
            assert proof["status"] == "original_sequences_qualified"
            assert proof["original_output_prefixes_match"] and proof["zero_major_faults"]
            assert proof["candidate_provider_sha256"] == build["provider_sha256"]
            assert proof["mode"] == mode and proof["trial"] == trial
            assert proof["full_validation"] == full and proof["context"] == args.context
            current = proof["graph_sha256"], proof["config_sha256"]
            identity = current if identity is None else identity
            assert identity == current
            for name, digest in proof["source_sha256"].items():
                assert sha(Path(name)) == digest, name
                sources.append(Path(name))
            assert [sample["warmup"] for sample in raw["samples"]] == [True, False]
            for sample in raw["samples"]:
                sequences[sample["warmup"]].add((sample["prompt_sha256"], tuple(sample["output_token_ids"])))
            sample = raw["samples"][1]
            trials.append(dict(trial=trial, selector=mode, ttft_seconds=sample["ttft_seconds"],
                               short_screen_decode_tps=sample["decode_tps"]))
            sources.append(proof_path)
        metrics = ttft_metrics([row["ttft_seconds"] for row in trials])
        rows.append(dict(full_validation=full, trials=trials, **metrics))
        print(json.dumps(rows[-1]), flush=True)
    assert all(len(values) == 1 for values in sequences.values())
    supported = all(row["ttft_win_screen_qualified"] for row in rows)
    report = dict(status="paired_ttft_win_both_modes" if supported else "held_no_paired_ttft_win_both_modes",
        context=args.context, provider_sha256=build["provider_sha256"],
        baseline_provider_sha256=build["baseline_provider_sha256"], graph_sha256=identity[0],
        config_sha256=identity[1], rows=rows, order=[0, 1, 1, 0],
        original_output_prefixes_match=True, repeated_sequences_match=True, zero_major_faults=True,
        short_screen_not_standard_decode=True, natural_256_token_decode_qualified=False,
        quality_qualified=False, default_enabled=False)
    execution.write_text(json.dumps(dict(status=report["status"], rows=rows), indent=2) + "\n")
    sources.append(execution)
    report["source_sha256"] = {str(path): sha(path) for path in sources}
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], context=args.context, rows=rows)), flush=True)


if __name__ == "__main__":
    main()
