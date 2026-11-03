import json
from pathlib import Path
import subprocess
import time

from qualify_quickgelu_mul_long_context import OVERLAY, ROOT, identities, load, sha, snapshot, validate
from recheck_quickgelu_mul_context_summary import SUMMARY, verify


def persist(report):
    SUMMARY.write_text(json.dumps(report, indent=2) + "\n")
    two = report["separate_2k_ABBA"]["candidate_metrics"]
    rows = f"| 2048 | {two['median_decode_tps']:.6f} | {two['aggregate_decode_tps']:.6f} | Separate ABBA candidate pool: 6 measured requests, 2 excluded warmups across 2 process blocks |"
    for context in ("8192", "32768"):
        if context in report["standard_rows"]:
            row = report["standard_rows"][context]
            rows += f"\n| {context} | {row['median_decode_tps']:.6f} | {row['aggregate_decode_tps']:.6f} | Standard standalone row: 3 measured requests, 1 excluded warmup |"
    identity = report["identity"]
    text = f"""# Same QuickGeluMul candidate: context qualification summary

Status: **{report['status']}**.

| Context tokens | Median decode tokens/s | Aggregate decode tokens/s | Timing design |
|---:|---:|---:|---|
{rows}

These are the **same immutable candidate graph/provider**, but **not identical
timing designs**. The 2K row is the candidate portion of the separate same-provider
ABBA experiment (six measured requests across two candidate blocks). The new8K
and prior32K rows are standalone standard runs, each with one excluded warmup
and three measured256-output requests,255 decode intervals each. Chunk16 and
pending-dispatch limit64 are unchanged; both exact-dense selectors remain enabled.

Every standard-row original prompt hash and complete256-token output sequence
is checked, including all warmups, against `int4-pending64-standard-contexts.json`.
Every phase major-fault count is zero. The separate2K ABBA token checks are also
revalidated. Data/config retain their original device/inode/size/mtime and
initializer descriptors. No model conversion, implementation, build, install,
weight copy or main-source integration was performed.

Provider SHA256: `{identity['provider_sha256']}`

Graph SHA256: `{identity['graph_sha256']}`

## Evidence and distinctions

- Original2K report/qualification are preserved:
  `report-resumed.md`, `qualification-resumed.json`.
- Prior32K report/qualification and benchmark evidence are preserved:
  `report-long-context-20261027.md`,
  `qualification-long-context-20261027.json`.
- Fresh8K evidence prefix:
  `{ROOT / 'quickgelu-mul-candidate-8k-followup-20261027'}`.
- This combined machine-readable summary:
  `{SUMMARY}`. It contains original inode records, provider/graph/source hashes,
  prior-evidence hashes, full benchmark/guard paths and initial/final ownership.

Old-provider8K/32K results are not rows in this table and are not matched controls.
There is **no causal long-context speedup, whole-model quality/logit equivalence,
integration or100TPS claim**. `quality_qualified=false`, `headline_eligible=false`.
The original2K ABBA caveats, including overlapping block ranges, remain applicable.

## Safety and read-only revalidation

Fresh all-eight-GPU snapshots and private-daemon readiness were checked. Only
idleGPU0 was selected under the existing no-foreign-owner/130000MiB guard.
GPU5/6/7 were not used. The existing networknone/user664503/capdropALL/
no-new-privileges runner and original guards were unchanged. Own GPU/container
cleanup: `{report.get('own_resources_released', False)}`.
The shared daemon remains available.

```bash
E={ROOT}
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={ROOT.parent / 'h200-native-combined-venv/lib/python3.12/site-packages'} \\
  python3 "$E/recheck_quickgelu_mul_context_summary.py"
```

Elapsed budget seconds: `{report.get('elapsed_seconds', 'running')}`.
Error/blocker: `{report.get('error', 'none')}`.
"""
    (OVERLAY / "report-context-summary-20261027.md").write_text(text)


def main():
    if SUMMARY.exists():
        raise FileExistsError(SUMMARY)
    started = time.monotonic()
    control = load(ROOT / "int4-pending64-standard-contexts.json")
    identity = identities(control)
    prior = load(OVERLAY / "qualification-long-context-20261027.json")
    assert prior["status"] == "32k_original_sequence_qualified_isolated_overlay"
    assert prior["identity"] == identity
    existing32 = prior["candidate_rows"]["32768"]
    assert validate(Path(existing32["benchmark"]), 32768, control, identity) == existing32
    two_path = OVERLAY / "qualification-resumed.json"
    two = load(two_path)
    preserve = [
        two_path, OVERLAY / "report-resumed.md",
        OVERLAY / "qualification-long-context-20261027.json",
        OVERLAY / "report-long-context-20261027.md", Path(existing32["benchmark"]),
        Path(existing32["guard"]), ROOT / "int4-pending64-standard-contexts.json",
    ]
    report = dict(status="running_8k_followup", identity=identity, standard_rows={"32768": existing32},
                  separate_2k_ABBA=dict(qualification=str(two_path),
                                       timing_design="separate same-provider ABBA candidate pool; not standalone",
                                       candidate_metrics=two["ABBA"]["summary"]["candidate"]),
                  preserved_prior_evidence_sha256={str(p): sha(p) for p in preserve},
                  initial_resources=snapshot(), quality_qualified=False, headline_eligible=False,
                  wall_budget_seconds=720, source_driver_sha256=sha(Path(__file__)))
    persist(report)
    try:
        prefix = ROOT / "quickgelu-mul-candidate-8k-followup-20261027"
        assert not any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".log"))
        invocation = load(Path(existing32["guard"]))["command"].copy()
        invocation[-1] = invocation[-1].replace("quickgelu-mul-candidate-32k-20261027", prefix.name).replace(
            "--contexts 32768", "--contexts 8192")
        remaining = 720 - (time.monotonic() - started)
        assert remaining > 500, "Insufficient bounded budget before starting8K"
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"),
                        "--gpu-index", "0", "--idle-seconds", "3", "--wait-seconds", "15",
                        "--run-seconds", str(int(remaining - 30)),
                        "--output-prefix", str(prefix), "--", *invocation], check=True)
        report["standard_rows"]["8192"] = validate(prefix.with_suffix(".json"), 8192, control, identity)
        assert identities(control) == identity
        report["status"] = "same_candidate_2k_abba_8k_32k_standard_qualified"
    except Exception as error:
        report["status"] = "blocked_or_failed_8k_preserving_prior_evidence"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["final_resources"] = snapshot()
        report["own_resources_released"] = (
            "GPU-c1593c4e" not in report["final_resources"]["compute_owners"] and
            not report["final_resources"]["private_containers"])
        persist(report)
    verify(report)
    print(json.dumps(report["standard_rows"]["8192"], indent=2), flush=True)
    print(SUMMARY, flush=True)


if __name__ == "__main__":
    main()
