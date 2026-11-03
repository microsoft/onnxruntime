import json
from pathlib import Path
import statistics
import subprocess


def main():
    root = Path(__file__).parent
    template = json.loads((root / "int4-shared-post-v4-candidate-2k-20261027.guard.json").read_text())
    original = json.loads((root / "int4-hot-engram-2k-20261026.json").read_text())
    rows = []
    for block, mode in enumerate(("control", "candidate", "candidate", "control")):
        prefix = root / f"quickgelu-mul-resumed-abba-{block}-{mode}-2k-20261027"
        if prefix.with_suffix(".guard.json").exists():
            raise FileExistsError(prefix)
        command = template["command"].copy()
        command[-1] = command[-1].replace(
            "/workspace/onnxruntime-webgpu-h200-perf/build-int4-shared-post-v4/",
            "/workspace/qwen38-int4-quickgelu-next/build/").replace(
            "int4-shared-post-v4-candidate-2k-20261027", prefix.name)
        if mode == "candidate":
            command[-1] = command[-1].replace(
                "/workspace/qwen38-int4-shared-post", "/workspace/qwen38-int4-quickgelu-next/model")
        subprocess.run(["python3", str(root / "shared_activation_guard.py"),
                        "--gpu-index", "1", "--idle-seconds", "3", "--wait-seconds", "20",
                        "--run-seconds", "300", "--output-prefix", str(prefix),
                        "--", *command], check=True)
        data = json.loads(prefix.with_suffix(".json").read_text())
        assert data["status"] == "completed" and not data["profiling"]
        assert len(data["samples"]) == 4
        for sample, reference in zip(data["samples"], original["samples"]):
            assert sample["prompt_sha256"] == reference["prompt_sha256"]
            assert len(sample["output_token_ids"]) == 256
            assert sample["output_token_ids"] == reference["output_token_ids"]
            assert all(p["major_page_faults"] == 0 for p in sample["process_resource_usage"].values())
        rows.append(dict(block=block, mode=mode, path=str(prefix.with_suffix(".json")),
                         median=data["summary"]["2048:configured"]["median_decode_tps"]))
        print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for mode in ("control", "candidate"):
        samples = [s for row in rows if row["mode"] == mode
                   for s in json.loads(Path(row["path"]).read_text())["samples"] if not s["warmup"]]
        summary[mode] = dict(median_decode_tps=statistics.median(s["decode_tps"] for s in samples),
                             aggregate_decode_tps=sum(s["output_tokens"] - 1 for s in samples) /
                             sum(s["decode_seconds"] for s in samples),
                             sample_decode_tps=[s["decode_tps"] for s in samples])
    report = dict(status="completed", order="control,candidate,candidate,control",
                  same_provider=True, rows=rows, summary=summary, original_sequences_exact=16,
                  quality_qualified=False, headline_eligible=False)
    (root / "quickgelu-mul-resumed-abba-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
