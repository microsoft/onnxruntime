import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import onnx

ROOT = Path(__file__).parent
REPOS = Path("/home/sayanshaw/repos")
OVERLAY = REPOS / "qwen38-int4-quickgelu-next"
QUALIFICATION = OVERLAY / "qualification-long-context-20261027.json"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def snapshot():
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_gpus": command(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
                             "--format=csv,noheader"]),
        "compute_owners": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                   "--format=csv,noheader"]),
        "private_containers": command(["sudo", "-n", "docker", "-H",
                                      "unix:///home/sayanshaw/docker-qwen38.sock",
                                      "ps", "--format", "{{.ID}} {{.Names}} {{.Status}}"]),
        "disk": command(["df", "-h", "/home/sayanshaw"]),
    }


def identities(control):
    prior = load(OVERLAY / "qualification-resumed.json")
    library = OVERLAY / "build/libonnxruntime_providers_webgpu.so"
    graph_path = OVERLAY / "model/model.onnx"
    original_path = REPOS / "qwen38-int4-webgpu-perf/model.onnx"
    assert sha(library) == prior["provider_sha256"]
    assert sha(graph_path) == prior["graph_sha256"]
    assert sha(original_path) == control["graph_sha256"]
    graph = onnx.load(graph_path, load_external_data=False)
    original = onnx.load(original_path, load_external_data=False)
    assert [v.SerializeToString() for v in graph.graph.initializer] == [
        v.SerializeToString() for v in original.graph.initializer]
    paths = {e.value for v in graph.graph.initializer for e in v.external_data
             if e.key == "location"} | {"genai_config.json"}
    inodes = {}
    for location in sorted(paths):
        candidate = graph_path.parent / location
        reference = original_path.parent / location
        assert candidate.samefile(reference), location
        info = reference.stat()
        inodes[location] = dict(device=info.st_dev, inode=info.st_ino,
                               size=info.st_size, mtime_ns=info.st_mtime_ns)
    assert sha(graph_path.parent / "genai_config.json") == control["config_sha256"]
    for path, digest in prior["source_sha256"].items():
        assert sha(Path(path.replace("/workspace/", str(REPOS) + "/"))) == digest, path
    return dict(provider_sha256=sha(library), graph_sha256=sha(graph_path),
                original_graph_sha256=sha(original_path), config_sha256=control["config_sha256"],
                source_sha256=prior["source_sha256"], unchanged_original_inodes=inodes,
                initializer_descriptors_unchanged=True)


def validate(path, context, control, identity):
    data, guard = load(path), load(path.with_suffix(".guard.json"))
    assert data["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
    assert not data["profiling"] and data["chunk"] == 16
    assert data["provider"] == "webgpu" and data["webgpu_max_pending_dispatches_override"] == 64
    assert guard["gpu_index"] == 0
    assert guard["gpu_uuid"] == "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd"
    for key in ("plugin_sha256", "graph_sha256", "config_sha256"):
        expected = identity["provider_sha256" if key == "plugin_sha256" else key]
        assert data[key] == expected, key
    assert not data["quality_qualified"] and not data["headline_eligible"]
    text = path.with_suffix(".log").read_text()
    assert "GetCapability()" not in text and "FAIL" not in text
    invocation = " ".join(guard["command"])
    assert "ORT_WEBGPU_HC_EXACT_DENSE=1" in invocation and "ORT_WEBGPU_EXACT_DENSE_MID=1" in invocation
    samples = data["samples"]
    references = [s for s in control["samples"] if s["context"] == context]
    assert len(samples) == len(references) == 4
    assert [s["warmup"] for s in samples] == [True, False, False, False]
    for sample, reference in zip(samples, references):
        for key in ("context", "repetition", "warmup", "seed", "offset", "input_tokens",
                    "prompt_sha256", "output_token_ids", "output_sha256"):
            assert sample[key] == reference[key], (context, key)
        assert sample["input_tokens"] == context
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
        assert sample["decode_seconds"] > 0
        assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
    measured = samples[1:]
    summary = data["summary"][f"{context}:configured"]
    return dict(context_tokens=context, median_decode_tps=summary["median_decode_tps"],
                aggregate_decode_tps=summary["decode_tps"],
                sample_decode_tps=[s["decode_tps"] for s in measured],
                sample_TTFT_seconds=[s["ttft_seconds"] for s in measured],
                excluded_warmups=1, measured_requests=3, outputs_each=256,
                timed_decode_intervals_each=255, original_sequences_exact_including_warmup=4,
                all_prompt_hashes_and_output_tokens_exact=True, all_major_page_faults_zero=True,
                benchmark=str(path), guard=str(path.with_suffix(".guard.json")),
                provider_sha256=data["plugin_sha256"], graph_sha256=data["graph_sha256"],
                guard_elapsed_seconds=guard["elapsed_seconds"], matched_control=False)


def write_report(report):
    rows = "\n".join(
        f"| {context} | {r['median_decode_tps']:.6f} | {r['aggregate_decode_tps']:.6f} | "
        f"{', '.join(f'{v:.3f}' for v in r['sample_decode_tps'])} |"
        for context, r in report["candidate_rows"].items())
    text = f"""# QuickGeluMul candidate long-context qualification

Status: **{report['status']}**.

## New candidate table — isolated overlay only

| Context tokens | Median decode tokens/s | Aggregate decode tokens/s | Three measured decode samples |
|---:|---:|---:|---|
{rows}

Each completed row uses the original standard protocol: one excluded warmup,
three measured 256-output requests, 255 timed decode intervals per request,
chunk16, pending-dispatch limit64, no profiling. Every original prompt hash and
every output token (including warmup) is checked against
`int4-pending64-standard-contexts.json`; all phase major-fault counts must be zero.
Both exact-dense selectors remain enabled. No tolerance or guard changes.

Provider SHA256: `{report['identity']['provider_sha256']}`

Graph SHA256: `{report['identity']['graph_sha256']}`

Data/config retain unchanged original device/inode/size/mtime and initializer
descriptors. Exact inode records and source hashes are in the adjacent JSON.
The existing 2K ABBA qualification remains separate and unchanged:
`qualification-resumed.json` / `report-resumed.md`.

## Scope and limits

This table contains **only the QuickGeluMul overlay provider**. Prior GDN/exact-mid
8K/32K rows use a different provider and are not candidate rows or matched timing
controls. In particular, the prior 32K median42.82323975 belongs to provider
`2a92e92a66dee76a83ceee23e834c5d0e2281a932a424f6ca62f6b6503b27f4c`;
no percentage improvement is computed against it.

There is **no matched-control or causal long-context speedup claim**, no whole-model
quality/logit-equivalence claim, no 100TPS claim and no main-source integration.
`quality_qualified=false`, `headline_eligible=false`.

8K scheduling: `{report.get('eight_k_status', 'not started')}`.

## Reproduction and cleanup

Driver/validator: `{ROOT / 'qualify_quickgelu_mul_long_context.py'}`.
The machine-readable qualification stores complete command/guard/report paths,
identity snapshots, original-reference hash, source hashes and final resource
ownership. Existing files are never overwritten by the measurement driver.

The original read-only NVML idle guard requires at least130000MiB free on the
selected GPU and no foreign compute owners. Only GPU0 is selected; GPU5/6/7 are
never used. The existing restrictive container runner retains networknone,
user664503:664503, capdropALL and no-new-privileges. No clocks/power changes,
weight copies, installs, builds, GitHub writes or existing-evidence edits.

Error/blocker: `{report.get('error', 'none')}`.
Final resources released: `{report.get('own_resources_released', False)}`.
Private daemon readiness was verified, and its existing shared service is left
available to the parent owner.
"""
    (OVERLAY / "report-long-context-20261027.md").write_text(text)


def main():
    if QUALIFICATION.exists():
        raise FileExistsError(QUALIFICATION)
    started = time.monotonic()
    deadline = started + 1800
    control_path = ROOT / "int4-pending64-standard-contexts.json"
    control = load(control_path)
    identity = identities(control)
    report = dict(status="running", candidate_rows={}, identity=identity,
                  original_reference=str(control_path), original_reference_sha256=sha(control_path),
                  initial_resources=snapshot(), quality_qualified=False, headline_eligible=False,
                  wall_budget_seconds=1800, existing_2k_qualification=str(OVERLAY / "qualification-resumed.json"),
                  source_driver_sha256=sha(Path(__file__)))

    def persist():
        QUALIFICATION.write_text(json.dumps(report, indent=2) + "\n")
        write_report(report)

    persist()
    try:
        template = load(ROOT / "int4-gdn-exact-mid-32k-20261027.guard.json")["command"]
        for context, label in ((32768, "32k"), (8192, "8k")):
            remaining = deadline - time.monotonic()
            if context == 8192:
                estimate = load(ROOT / "int4-gdn-exact-mid-8k-20261027.guard.json")["elapsed_seconds"]
                if remaining < estimate + 60:
                    report["eight_k_status"] = (
                        f"skipped: {remaining:.1f}s budget remains, prior8K runtime={estimate:.1f}s plus60s margin")
                    persist()
                    break
                report["eight_k_status"] = "running"
            prefix = ROOT / f"quickgelu-mul-candidate-{label}-20261027"
            assert not any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".log"))
            invocation = template.copy()
            invocation[-1] = invocation[-1].replace(
                "/workspace/qwen38-int4-gdn-prep", "/workspace/qwen38-int4-quickgelu-next/model").replace(
                "/workspace/qwen38-int4-dense-next/followup2560/build/",
                "/workspace/qwen38-int4-quickgelu-next/build/").replace(
                "int4-gdn-exact-mid-32k-20261027", prefix.name).replace(
                "--contexts 32768", f"--contexts {context}")
            report["active_context"] = context
            persist()
            subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"),
                            "--gpu-index", "0", "--idle-seconds", "3", "--wait-seconds", "15",
                            "--run-seconds", str(max(0, int(remaining - 25))),
                            "--output-prefix", str(prefix), "--", *invocation], check=True)
            report["candidate_rows"][str(context)] = validate(
                prefix.with_suffix(".json"), context, control, identity)
            assert identities(control) == identity
            if context == 8192:
                report["eight_k_status"] = "completed and original-sequence-qualified"
            persist()
            print(json.dumps(report["candidate_rows"][str(context)]), flush=True)
        report["status"] = ("32k_original_sequence_qualified_isolated_overlay"
                            if "32768" in report["candidate_rows"] else "blocked_no_qualified_32k_row")
    except Exception as error:
        report["status"] = "blocked_or_failed_preserving_partial_evidence"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["final_resources"] = snapshot()
        owners = report["final_resources"]["compute_owners"]
        report["own_resources_released"] = "GPU-c1593c4e" not in owners
        report.pop("active_context", None)
        persist()
        print(QUALIFICATION, flush=True)


if __name__ == "__main__":
    main()
