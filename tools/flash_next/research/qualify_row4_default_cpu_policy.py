import argparse
import json
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT, sha
from qualify_row4_cpu_locality import compare_samples
from run_row4_default_cpu_policy import STAMP, VARIANT, prefix_for, verify_cpu_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    phase = "repeat-screen" if args.repeat else "screen"
    output = ROOT / f"{VARIANT}-context{args.context}-{phase}-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    sources, rows = {}, []
    provider = None
    initial_provider = None
    trial = int(args.repeat)
    if args.repeat:
        path = ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["status"] == "initial_default_policy_decode_gain_both_modes"
        initial_provider = proof["provider_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(ROOT / name) == digest, name
            sources[name] = digest
        sources[path.name] = sha(path)
    for full in (True, False):
        samples, results, times = [], [], []
        for placement, selected_trial in (("baseline", trial), ("local8", trial), ("baseline", trial + 1)):
            prefix = prefix_for(placement, args.context, full, selected_trial)
            proof_path = prefix.with_suffix(".qualification.json")
            raw_path = prefix.with_suffix(".json")
            guard_path = prefix.with_suffix(".guard.json")
            proof, raw, guard = (json.loads(path.read_text()) for path in (proof_path, raw_path, guard_path))
            assert proof["status"] == "original_sequences_qualified"
            assert proof["zero_major_faults"] and proof["original_output_prefixes_match"]
            assert proof["placement"] == placement and proof["trial"] == selected_trial
            assert proof["context"] == args.context and proof["full_validation"] == full
            assert proof["default_worker_policy_preserved"] == (placement == "baseline")
            verify_cpu_policy(raw, placement)
            provider = proof["provider_sha256"] if provider is None else provider
            assert proof["provider_sha256"] == provider
            if initial_provider is not None:
                assert provider == initial_provider
            for name, digest in proof["source_sha256"].items():
                assert sha(ROOT / name) == digest, name
                sources[name] = digest
            sources[proof_path.name] = sha(proof_path)
            measured = [sample for sample in raw["samples"] if not sample["warmup"]]
            assert len(raw["samples"]) == 4 and len(measured) == 3
            assert guard["status"] == "completed" and guard["exit_code"] == 0
            samples.append(measured)
            results.append(proof["result"])
            times.append((proof["started_ns"], proof["completed_ns"]))
        assert times[0][1] <= times[1][0] and times[1][1] <= times[2][0], times
        rows.append(dict(full_validation=full, before=results[0], candidate=results[1],
                         after=results[2], comparison=compare_samples(samples[1], samples[0], samples[2])))
    supported = all(row["comparison"]["supported"] for row in rows)
    report = dict(status=("repeated_default_policy_decode_gain_both_modes" if args.repeat else
                          "initial_default_policy_decode_gain_both_modes") if supported else "no_supported_decode_gain",
                  context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,
                  default_worker_policy_preserved_in_controls=True, original_output_prefixes_match=True,
                  zero_major_faults=True, process_only_affinity=True, memory_policy_unchanged=True,
                  comparison_kind="bracketed_separate_load_default_worker_policy_screen_not_kernel_gain",
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={**sources, Path(__file__).name: sha(Path(__file__)),
                                 "qualify_row4_cpu_locality.py": sha(ROOT / "qualify_row4_cpu_locality.py")})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "repeat", "rows")}), flush=True)


if __name__ == "__main__":
    main()
