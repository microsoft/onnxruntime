import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    qualification = load(root / "gdn-prep-qualification.json")
    control = load(root / "int4-pending64-standard-contexts.json")
    model = Path("/home/sayanshaw/repos/qwen38-int4-gdn-prep/model.onnx")
    library = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/build-int4-gdn-prep/"
                   "libonnxruntime_providers_webgpu.so")
    if digest(model) != qualification["graph_sha256"] or digest(library) != qualification["provider_sha256"]:
        raise ValueError("Qualified graph/provider changed")
    rows = {}
    for path in args.reports:
        data = load(path)
        guard = load(path.with_suffix(".guard.json"))
        if (data["status"] != "completed" or guard["status"] != "completed" or guard["exit_code"] != 0 or
                data["profiling"] or data["plugin_sha256"] != qualification["provider_sha256"] or
                data["graph_sha256"] != qualification["graph_sha256"] or
                data["config_sha256"] != control["config_sha256"] or
                data["webgpu_max_pending_dispatches_override"] != 64):
            raise ValueError(f"Unqualified benchmark protocol or identities: {path}")
        contexts = {s["context"] for s in data["samples"]}
        for context in contexts:
            if str(context) in rows:
                raise ValueError(f"Duplicate context {context}")
            samples = [s for s in data["samples"] if s["context"] == context]
            reference = [s for s in control["samples"] if s["context"] == context]
            if len(samples) != 4 or len(reference) != 4 or sum(s["warmup"] for s in samples) != 1:
                raise ValueError(f"Expected excluded warmup plus three measured requests: {context}")
            for sample, original in zip(samples, reference):
                if (sample["output_tokens"] != 256 or len(sample["output_token_ids"]) != 256 or
                        sample["warmup"] != original["warmup"] or
                        sample["prompt_sha256"] != original["prompt_sha256"] or
                        sample["output_token_ids"] != original["output_token_ids"]):
                    raise ValueError(f"Full original sequence parity failed: {context}")
            measured = [s for s in samples if not s["warmup"]]
            summary = data["summary"][f"{context}:configured"]
            rows[str(context)] = {
                "median_decode_tps": summary["median_decode_tps"],
                "aggregate_decode_tps": summary["decode_tps"],
                "sample_decode_tps": [s["decode_tps"] for s in measured],
                "sample_TTFT_seconds": [s["ttft_seconds"] for s in measured],
                "measured_requests": 3, "outputs_each": 256, "excluded_warmups": 1,
                "timed_decode_intervals_each": 255,
                "exact_original_graph_sequences_including_warmup": 4,
                "process_major_page_faults": [
                    {phase: usage["major_page_faults"] for phase, usage in s["process_resource_usage"].items()}
                    for s in samples],
                "report": str(path),
            }
    report = {
        "status": "native_and_original_sequence_qualified_not_whole_model_quality_qualified",
        "provider_sha256": qualification["provider_sha256"],
        "graph_sha256": qualification["graph_sha256"],
        "config_sha256": control["config_sha256"],
        "rows": rows,
        "complete_2k_8k_32k_table": set(rows) == {"2048", "8192", "32768"},
        "headline_eligible": False, "quality_qualified": False,
        "caveats": [
            "Sequential standalone timings, not a counterbalanced causal speedup",
            "Preserves original-model sequences but not whole-model numerical-quality qualification",
            "Original graph, weights and configuration remain unchanged; explicit fusion derivative only",
            "Exact dense optimization is a separate candidate, not present in this provider or these rows",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
