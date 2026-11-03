import collections
import json
from pathlib import Path
import re

from build_qmoe_node_replay import sha
from run_decode_gpu_tail_profile_v1 import gpu_events
from run_gpu_slice_model import verify_sources


ROOT = Path(__file__).parent
WORKGROUPS = {"QmoeHierarchicalGate1Token": 512, "QmoeFC1SwiGLUBitcastF16": 128,
              "QmoeFC2FinalMixBitcastF16": 320}


def summarize(profiles):
    if len(profiles) != 4:
        raise RuntimeError("Expected exactly four measured token profiles")
    total, counts = collections.Counter(), collections.Counter()
    scope_counts = []
    for profile in profiles:
        events = gpu_events(profile)
        scope_counts.append(len(events))
        local = collections.Counter()
        for event in events:
            name = event["name"].split("&")[-1]
            local[name] += 1
            counts[name] += 1
            total[name] += event["dur"]
            if name in WORKGROUPS:
                match = re.search(r"\]:(\d+),(\d+),(\d+):", event.get("args", {}).get("cache_key", ""))
                if match is None or tuple(map(int, match.groups())) != (WORKGROUPS[name], 1, 1):
                    raise RuntimeError(f"Incorrect deployed workgroup for {name}")
        if any(local[name] != 48 for name in WORKGROUPS):
            raise RuntimeError("Missing complete 48-layer deployed QMoE routing")
    if len(set(scope_counts)) != 1:
        raise RuntimeError("Measured token profiles have different GPU dispatch counts")
    return {
        "measured_token_profiles": 4,
        "gpu_events_per_token": scope_counts[0],
        "summed_instrumented_gpu_us_per_token": sum(total.values()) / 4,
        "program_gpu_us_per_token": {name: value / 4 for name, value in total.items()},
        "program_dispatches_per_token": {name: value / 4 for name, value in counts.items()},
    }


def main():
    output = ROOT / "decode-gpu-context-2k-32k-v1.json"
    if output.exists():
        raise FileExistsError(output)
    evidence = [Path(__file__), ROOT / "run_decode_gpu_tail_profile_v1.py"]
    summaries, sources = {}, {}
    for context, filename, status in (
            (2048, "decode-gpu-tail-profile-v1-outcome.json", "decode_gpu_tail_profile_completed"),
            (32768, "decode-gpu-tail-profile-32k-v1-outcome.json", "decode_gpu_tail_profile_32k_completed")):
        path = ROOT / filename
        outcome = verify_sources(path, status)
        if (outcome["context"] != context or not outcome["diagnostic_only"]
                or not outcome["model_original_prefixes_agree"] or outcome["performance_qualified"]):
            raise RuntimeError("Wrong or unverified diagnostic scope")
        profiles = []
        for record in outcome["profiles"]:
            if record["warmup"]:
                continue
            file = Path(record["file"])
            events = json.loads(file.read_text())
            actual = gpu_events(events)
            if (len(actual) != record["gpu_events"]
                    or sum(event["dur"] for event in actual) != record["summed_gpu_us"]):
                raise RuntimeError("Saved profile metadata differs from actual GPU events")
            profiles.append(events)
        summaries[str(context)] = summarize(profiles)
        evidence.append(path)
        raw_path = ROOT / filename.replace("-outcome.json", ".json")
        raw = json.loads(raw_path.read_text())
        sources[str(context)] = {key: raw[key] for key in
                                ("genai_native", "plugin_sha256", "graph_sha256",
                                 "config_sha256", "load_experiment_flags")}
        evidence.append(raw_path)
    if sources["2048"] != sources["32768"]:
        raise RuntimeError("Context profiles have different model/runtime/provider/load flags")
    short = summaries["2048"]["program_gpu_us_per_token"]
    long = summaries["32768"]["program_gpu_us_per_token"]
    deltas = sorted(({"program": name, "context2k_us": short.get(name, 0),
                      "context32k_us": long.get(name, 0),
                      "delta_us": long.get(name, 0) - short.get(name, 0)}
                     for name in short.keys() | long.keys()), key=lambda row: row["delta_us"], reverse=True)
    report = {
        "status": "decode_gpu_context_diagnostics_compared",
        "diagnostic_only": True,
        "performance_qualified": False,
        "quality_qualified": False,
        "scope": "instrumented native GPU kernel sums, not natural latency/TPS or predicted gains",
        "same_model_runtime_provider_and_load_flags": True,
        "summaries": summaries,
        "program_deltas": deltas,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"summaries": {key: {k: v for k, v in value.items()
                                         if not k.startswith("program_")} for key, value in summaries.items()},
                      "largest_context_deltas": deltas[:10]}), flush=True)


if __name__ == "__main__":
    main()
