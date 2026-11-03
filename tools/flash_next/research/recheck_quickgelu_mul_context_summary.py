import math
from pathlib import Path

from qualify_quickgelu_mul_long_context import OVERLAY, ROOT, identities, load, sha, validate

SUMMARY = OVERLAY / "qualification-context-summary-20261027.json"


def verify(report):
    assert report["status"] == "same_candidate_2k_abba_8k_32k_standard_qualified"
    assert not report["quality_qualified"] and not report["headline_eligible"]
    assert sha(Path(__file__).with_name("finish_quickgelu_mul_8k.py")) == report["source_driver_sha256"]
    control = load(ROOT / "int4-pending64-standard-contexts.json")
    assert identities(control) == report["identity"]
    for path, digest in report["preserved_prior_evidence_sha256"].items():
        assert sha(Path(path)) == digest, path
    for context in ("8192", "32768"):
        row = report["standard_rows"][context]
        assert validate(Path(row["benchmark"]), int(context), control, report["identity"]) == row
    two = load(Path(report["separate_2k_ABBA"]["qualification"]))
    original = load(ROOT / "int4-hot-engram-2k-20261026.json")
    assert two["provider_sha256"] == report["identity"]["provider_sha256"]
    assert two["graph_sha256"] == report["identity"]["graph_sha256"]
    assert two["ABBA"]["status"] == "completed" and two["ABBA"]["same_provider"]
    assert [r["mode"] for r in two["ABBA"]["rows"]] == ["control", "candidate", "candidate", "control"]
    assert report["separate_2k_ABBA"]["candidate_metrics"] == two["ABBA"]["summary"]["candidate"]
    for block in two["ABBA"]["rows"]:
        data = load(Path(block["path"]))
        assert data["status"] == "completed" and data["plugin_sha256"] == two["provider_sha256"]
        assert data["config_sha256"] == report["identity"]["config_sha256"]
        assert not data["profiling"] and data["webgpu_max_pending_dispatches_override"] == 64
        assert [s["warmup"] for s in data["samples"]] == [True, False, False, False]
        if block["mode"] == "candidate":
            assert data["graph_sha256"] == two["graph_sha256"]
        for sample, reference in zip(data["samples"], original["samples"]):
            assert sample["prompt_sha256"] == reference["prompt_sha256"]
            assert sample["output_token_ids"] == reference["output_token_ids"]
            assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
            assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
            assert all(p["major_page_faults"] == 0 for p in sample["process_resource_usage"].values())
    assert report["own_resources_released"]
    assert report["final_resources"]["private_containers"] == ""
    assert "GPU-c1593c4e" not in report["final_resources"]["compute_owners"]


if __name__ == "__main__":
    verify(load(SUMMARY))
    print("PASS immutable same candidate: separate2K ABBA and standard8K/32K original sequences")
    print("PASS original hashes/inodes/descriptors, zero phase major faults and recorded own-resource cleanup")
