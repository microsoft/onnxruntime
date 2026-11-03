from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main


def verify_staged_stats(rows):
    names = {"HCPostNormProjectionF16", "HCSharedPostNormProjectionF16",
             "HCStagedNorm", "HCSharedStagedNorm", "HCStagedProjection", "HCSharedStagedProjection"}
    assert len(rows) == 36
    assert all(sum(r["operator"] == name for r in rows) == 6 for name in names)
    assert all(r["operator"] in names and r["requests"] == 96 and
               r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
               r["hits"] + r["captures"] == r["requests"] and r["evictions"] > 0 and r["captures"] >= 65 and
               0 < r["entries"] <= r["capacity"] == 64 and
               0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-staged-projection/build-v1"),
         prefix=ROOT / "hc-staged-projection-v1-native-gpu7-20261030",
         report_status="hc_staged_projection_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3/build-manifest.json"),
         stats_verifier=verify_staged_stats)
