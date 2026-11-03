import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REPOS = Path("/home/sayanshaw/repos")
PREFIX = ROOT / "int4-mlp-full-model-v2"
QUALITY = ROOT / (PREFIX.name + "-quality.json")
STATE = ROOT / "int4-mlp-full-model-queue.json"


def persist(**values):
    STATE.write_text(json.dumps(values, indent=2) + "\n")


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"Incomplete JSON while reading {path}; attempt {attempt + 1}/3", flush=True)
            if attempt == 2:
                raise
            time.sleep(1)


def context_table(result):
    if result["profiling"]:
        raise RuntimeError("Profiled measurements cannot populate the unprofiled throughput table")
    rows = []
    for context in (2048, 8192, 32768):
        mode = "control" if context == 2048 else "configured"
        samples = [r for r in result["samples"]
                   if r["context"] == context and r["mode"] == mode and not r["warmup"]]
        if len(samples) > 3:
            raise RuntimeError(f"Unexpected request count for context {context}")
        for sample in samples:
            if sample["input_tokens"] != context or sample["output_tokens"] != 256:
                raise RuntimeError("Unexpected context-table input/output token counts")
            if any(not math.isfinite(sample[k]) or sample[k] <= 0
                   for k in ("ttft_seconds", "decode_tps")):
                raise RuntimeError("Context-table timing metrics must be finite and positive")
        complete = len(samples) == 3
        if complete and {r["repetition"] for r in samples} != {0, 1, 2}:
            raise RuntimeError("Context-table repetition identities differ")
        ttft = statistics.median(r["ttft_seconds"] for r in samples) if complete else None
        rows.append({
            "context_tokens": context, "measured_requests": len(samples),
            "status": "completed" if complete else "partial" if samples else "pending",
            "median_ttft_seconds": ttft,
            "prompt_tokens_per_ttft_second": context / ttft if complete else None,
            "median_decode_tps": statistics.median(r["decode_tps"] for r in samples) if complete else None,
        })
    return rows


def publish_context_table():
    source = ROOT / "int4-split-long-context-skinny-protocol.json"
    result = read_json(source)
    rows = context_table(result)
    summary = {
        "source": str(source), "source_protocol_status": result["status"], "rows": rows,
        "all_contexts_complete": all(r["status"] == "completed" for r in rows),
        "configuration": "Original INT4, split-KV=1, uniform arena=1, skinny=0, chunk16,256 outputs",
        "scope": "2K control requests and8K/32K configured requests from one model load; warmups excluded",
        "quality_qualified": False, "headline_eligible": False,
        "caveat": "Prompt tokens / TTFT includes first-token latency; not isolated prefill GPU throughput. "
                  "Partial contexts have no reported median. Different prompts/generated tokens may change routing.",
    }
    (ROOT / "int4-split-long-completed-contexts-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with (ROOT / "int4-split-long-context-table.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    predecessor = ROOT / "int4-split-long-context-skinny-protocol.guard.json"
    deadline = time.monotonic() + 21600
    persist(status="waiting_for_previous_model_release", predecessor=str(predecessor))
    while True:
        previous = read_json(predecessor)
        publish_context_table()
        if previous["status"] in {"completed", "failed", "run_timed_out", "yielded_to_training"}:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Previous model protocol did not finish within the queue deadline")
        time.sleep(30)
    if previous["status"] != "completed" or previous.get("exit_code") != 0:
        print(f"Previous protocol ended {previous['status']}; preserve its incomplete results. "
              "The next model still requires an independently idle GPU.", flush=True)

    expected = {
        REPOS / "qwen38-int4-webgpu-perf/model.onnx":
            "fd3a152c8f4dbf93aae8d12060d16b0714dc5977f1d8c7e3eea19098f9268426",
        REPOS / "qwen38-int4-webgpu-perf/genai_config.json":
            "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6",
        REPOS / "qwen38-int4-webgpu-perf-mlp-fused/model.onnx":
            "bd6d338361552f31469fd7d845b3c99b5fe1b582ae54251b288e083ade19579c",
        REPOS / "onnxruntime-webgpu-h200-perf/build-int4-dsa-split-kv/libonnxruntime_providers_webgpu.so":
            "bd08a5cfe7eee2d9aad0bbd6be84a5f228f0b806e0e31b7538c95f5d53bb61ab",
        REPOS / "onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release/libonnxruntime.so.1.32.0":
            "7a4b7503353a7a5fc5256727f4f06ede3b2807cb1a15c7ec530d1b5746ed7f95",
        REPOS / "onnxruntime-genai-webgpu-fp8-checkpoint/build-h200-int4-legacy/libonnxruntime-genai.so":
            "0697af07edc986545f6a54c3649be02af116aba616dc71292cd46ae121a7faa9",
    }
    for path, digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen input changed: {path}")
    for suffix in (".json", ".guard.json", ".log"):
        if PREFIX.with_suffix(suffix).exists():
            raise FileExistsError(PREFIX.with_suffix(suffix))
    command = f"""set -e
export LD_LIBRARY_PATH=/workspace/onnxruntime-genai-webgpu-fp8-checkpoint/build-h200-int4-legacy:/workspace/qwen38-fp8-webgpu-runtime/sdk/lib:/workspace/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release:/workspace/onnxruntime-genai-webgpu-fp8-checkpoint/build-webgpu-fp8/Release
/workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd
export ORT_WEBGPU_QMOE_INT4_PREFILL=1 ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL=1 ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER=1 ORT_WEBGPU_SMALL_READBACK_CACHE=1 ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE=1 ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_UNIFORM_ARENA=1 ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP=0 ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV=1
exec python3 -u /evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py --model /workspace/qwen38-int4-webgpu-perf-mlp-fused --plugin /workspace/onnxruntime-webgpu-h200-perf/build-int4-dsa-split-kv/libonnxruntime_providers_webgpu.so --output /evidence/webgpu-perf-oct20/{PREFIX.name}.json --quality-output /evidence/webgpu-perf-oct20/{PREFIX.name}-quality.json --contexts 2048 --tokens 256 --repetitions 3
"""
    persist(status="starting_guarded_candidate", predecessor_status=previous["status"])
    result = subprocess.run([
        sys.executable, str(ROOT / "safe_model_run.py"), "--gpu-index", "0",
        "--idle-seconds", "5", "--wait-seconds", "60", "--run-seconds", "3600",
        "--output-prefix", str(PREFIX), "--", "bash", "-lc", command,
    ])
    if result.returncode:
        raise RuntimeError(f"Candidate guard failed with exit code {result.returncode}")
    candidate = read_json(PREFIX.with_suffix(".json"))
    quality = read_json(QUALITY)
    if candidate["status"] != "completed" or quality["status"] != "completed":
        raise RuntimeError("Candidate guard exited without completed model and sanity-check evidence")
    control = read_json(ROOT / "int4-split-long-context-skinny-protocol.json")
    for field in ("plugin_sha256", "config_sha256"):
        if candidate[field] != control[field]:
            raise RuntimeError(f"Candidate runtime identity differs: {field}")
    for field in ("genai_native", "ort_native"):
        if candidate[field]["sha256"] != control[field]["sha256"]:
            raise RuntimeError(f"Candidate loaded a different native library: {field}")
    baseline = {r["repetition"]: r for r in control["samples"]
                if r["context"] == 2048 and r["mode"] == "control" and not r["warmup"]}
    measured = [r for r in candidate["samples"] if not r["warmup"]]
    if len(baseline) != 3 or len(measured) != 3:
        raise RuntimeError("Expected three original and three candidate measured requests")
    if {r["repetition"] for r in measured} != set(baseline):
        raise RuntimeError("Measured repetition identities differ")
    pairs = []
    for row in measured:
        before = baseline[row["repetition"]]
        if row["context"] != 2048 or row["output_tokens"] != 256 or before["output_tokens"] != 256:
            raise RuntimeError("Unexpected candidate context or generated-token count")
        if row["prompt_sha256"] != before["prompt_sha256"]:
            raise RuntimeError("Original and candidate prompt identities differ")
        pairs.append({
            "repetition": row["repetition"],
            "control_tps": before["decode_tps"], "candidate_tps": row["decode_tps"],
            "control_ttft_seconds": before["ttft_seconds"],
            "candidate_ttft_seconds": row["ttft_seconds"],
            "output_tokens_match": before["output_token_ids"] == row["output_token_ids"],
            "first_different_token": next(
                (i for i, (a, b) in enumerate(zip(before["output_token_ids"], row["output_token_ids"]))
                 if a != b), None),
        })
    before = statistics.median(p["control_tps"] for p in pairs)
    after = statistics.median(p["candidate_tps"] for p in pairs)
    summary = {
        "status": "completed", "context": 2048, "tokens_per_request": 256,
        "comparison_design": "Serial model loads, same seeded prompts; not counterbalanced loads",
        "pairs": pairs, "control_median_tps": before, "candidate_median_tps": after,
        "observed_ratio": after / before, "quality_qualified": False, "headline_eligible": False,
        "quality_evidence": str(QUALITY),
        "candidate_evidence": str(PREFIX.with_suffix(".json")),
        "prior_long_protocol_status": previous["status"],
        "caveat": "Operator-qualified original-weight fusion; no identical-routing or quality claim.",
    }
    (ROOT / "int4-mlp-full-model-comparison-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    persist(status="completed", comparison_summary=str(ROOT / "int4-mlp-full-model-comparison-summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        persist(status="failed", error=f"{type(error).__name__}: {error}")
        raise
