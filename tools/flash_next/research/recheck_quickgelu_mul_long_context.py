from pathlib import Path

from qualify_quickgelu_mul_long_context import (
    OVERLAY, QUALIFICATION, identities, load, sha, validate,
)

report = load(QUALIFICATION)
assert report["status"] == "32k_original_sequence_qualified_isolated_overlay"
assert not report["quality_qualified"] and not report["headline_eligible"]
reference = Path(report["original_reference"])
assert sha(reference) == report["original_reference_sha256"]
control = load(reference)
assert identities(control) == report["identity"]
driver = Path(__file__).with_name("qualify_quickgelu_mul_long_context.py")
assert sha(driver) == report["source_driver_sha256"]
for context, row in report["candidate_rows"].items():
    assert validate(Path(row["benchmark"]), int(context), control, report["identity"]) == row
assert "32768" in report["candidate_rows"]
assert report["own_resources_released"]
assert report["final_resources"]["private_containers"] == ""
assert "GPU-c1593c4e" not in report["final_resources"]["compute_owners"]
assert (OVERLAY / "report-long-context-20261027.md").is_file()
print("PASS immutable candidate hashes/inodes, original complete sequences, 255 intervals, zero major faults")
print("PASS recorded own-resource cleanup; no long-context causal speedup/quality claim")
