from pathlib import Path

from run_hc_bitcast_native import main
from qualify_quickgelu_mul_long_context import ROOT


def verify_decoded_stats(rows):
    names = {"HCPostNormProjectionF16", "HCSharedPostNormProjectionF16",
             "HCPostNormProjectionPredecodedF16", "HCSharedPostNormProjectionPredecodedF16"}
    assert len(rows) == 24
    assert all(sum(r["operator"] == name for r in rows) == 6 for name in names)
    assert all(r["operator"] in names and r["requests"] == 96 and
               r["rebinds"] + r["binding_hits"] == r["hits"] and r["binding_hits"] > 0 and
               r["hits"] + r["captures"] == r["requests"] and r["evictions"] > 0 and r["captures"] >= 65 and
               0 < r["entries"] <= r["capacity"] == 64 and
               0 < r["binding_entries"] <= r["binding_capacity"] == 64 for r in rows), rows


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-predecode/build-v2"),
         prefix=ROOT / "hc-post-norm-predecode-v2-native-gpu7-20261030",
         report_status="hc_post_norm_predecode_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v2/build-manifest.json"),
         stats_verifier=verify_decoded_stats)
