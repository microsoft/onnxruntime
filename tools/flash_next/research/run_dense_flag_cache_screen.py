import argparse
import json
from pathlib import Path
import subprocess
import time

from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_row4_cpu_locality import compare_samples
from run_dense_flag_cache_model import FLAG, VARIANT, prefix_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    name = f"{VARIANT}-context{args.context}-{'repeat-' if args.repeat else ''}screen-20261031"
    manifest_path = ROOT / (name + "-execution.json")
    output = ROOT / (name + ".qualification.json")
    if manifest_path.exists() or output.exists():
        raise FileExistsError(name)
    sources = {str(Path(__file__)): sha(Path(__file__))}
    initial = None
    records = []
    if args.repeat:
        path = ROOT / f"{VARIANT}-context{args.context}-screen-20261031.qualification.json"
        initial = json.loads(path.read_text())
        assert initial["status"] == "initial_cache_decode_gain_both_modes"
        for source, digest in initial["source_sha256"].items():
            assert sha(Path(source)) == digest, source
        sources[str(path)] = sha(path)
        prior_manifest = ROOT / f"{VARIANT}-context{args.context}-screen-20261031-execution.json"
        records.extend(json.loads(prior_manifest.read_text())["records"])
    manifest = dict(status="running", records=records, context=args.context, repeat=args.repeat,
                    driver_sha256=sha(Path(__file__)))
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    trial = int(args.repeat)
    for full in (True, False):
        for mode, index in (("control", trial), ("cache", trial), ("control", trial + 1)):
            prefix = prefix_for(mode, args.context, full, index)
            if args.repeat and mode == "control" and index == trial:
                assert any(record["prefix"] == str(prefix) and record["completed"]
                           for record in records)
                continue
            command = ["python3", str(ROOT / "run_dense_flag_cache_model.py"),
                       "--mode", mode, "--context", str(args.context), "--trial", str(index)]
            if full:
                command.append("--full-validation")
            record = dict(prefix=str(prefix), command=command, started_ns=time.time_ns(), completed=False)
            records.append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            try:
                subprocess.run(command, check=True)
                record["proof_sha256"] = sha(prefix.with_suffix(".qualification.json"))
                record["completed"] = True
            finally:
                record["completed_ns"] = time.time_ns()
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    providers = {}
    for full in (True, False):
        samples, results, times = [], [], []
        for mode, index in (("control", trial), ("cache", trial), ("control", trial + 1)):
            prefix = prefix_for(mode, args.context, full, index)
            record = next(r for r in records if r["prefix"] == str(prefix))
            assert record["completed"]
            proof_path = prefix.with_suffix(".qualification.json")
            assert sha(proof_path) == record["proof_sha256"]
            proof, raw, guard = (json.loads(prefix.with_suffix(s).read_text())
                                 for s in (".qualification.json", ".json", ".guard.json"))
            assert proof["status"] == "original_sequences_qualified"
            assert proof["phase"] == "standard" and proof["mode"] == mode and proof["trial"] == index
            assert proof["context"] == args.context and proof["full_validation"] == full
            assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
            assert proof["worker_policy_unchanged"] and raw["cpu_affinity"] == list(range(96))
            assert proof["experiment_flag"] == FLAG and proof["experiment_flag_value"] == int(mode == "cache")
            assert guard["status"] == "completed" and guard["exit_code"] == 0
            assert f"export {FLAG}={int(mode == 'cache')}; " in guard["command"][-1]
            providers.setdefault(mode, proof["provider_sha256"])
            assert proof["provider_sha256"] == raw["plugin_sha256"] == providers[mode]
            assert proof["original_provider_control"] == (mode == "control")
            for source, digest in proof["source_sha256"].items():
                assert sha(Path(source)) == digest, source
                sources[source] = digest
            sources[str(proof_path)] = sha(proof_path)
            measured = [sample for sample in raw["samples"] if not sample["warmup"]]
            assert len(raw["samples"]) == 4 and len(measured) == 3
            for sample in measured:
                assert sample["output_tokens"] == 256
                assert abs(sample["decode_tps"] * sample["decode_seconds"] - 255) < 1e-8
            samples.append(measured)
            results.append(proof["result"])
            times.append((record["started_ns"], record["completed_ns"]))
        assert times[0][1] <= times[1][0] and times[1][1] <= times[2][0]
        rows.append(dict(full_validation=full, before=results[0], candidate=results[1],
                         after=results[2], comparison=compare_samples(samples[1], samples[0], samples[2])))
    assert providers["control"] != providers["cache"]
    assert providers["control"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"
    provider = providers["cache"]
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sources[str(manifest_path)] = sha(manifest_path)
    supported = all(row["comparison"]["supported"] for row in rows)
    report = dict(status=(("repeated_" if args.repeat else "initial_") + "cache_decode_gain_both_modes"
                          if supported else "no_supported_decode_gain"),
                  context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,
                  original_baseline_provider_sha256=providers["control"],
                  all_candidate_modes_above_100_decode_tps=all(
                      row["candidate"]["median_decode_tps"] > 100 for row in rows),
                  original_output_prefixes_match=True, zero_major_faults=True, worker_policy_unchanged=True,
                  comparison_kind="bracketed_separate_load_original_provider_net_screen",
                  quality_qualified=False, headline_eligible=False, source_sha256=sources)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], context=args.context, repeat=args.repeat,
        results=[dict(full=r["full_validation"], tps=r["candidate"]["median_decode_tps"],
                      ttft=r["candidate"]["mean_ttft_seconds"], comparison=r["comparison"])
                 for r in rows])), flush=True)


if __name__ == "__main__":
    main()
