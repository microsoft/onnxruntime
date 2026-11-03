import argparse
import collections
from pathlib import Path

from build_dense_qkv_predecode8 import STATS
from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main


def verify_stats(rows):
    assert collections.Counter(row["operator"] for row in rows) == {
        "MatMulNBitsExact": 9, STATS: 1}
    for row in rows:
        assert row["requests"] in ((128,) if row["operator"] == STATS else (128, 256))
        assert row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0
        assert row["hits"] + row["captures"] == row["requests"]
        assert row["entries"] == 1 and row["capacity"] == 64
        assert 0 < row["binding_entries"] <= row["binding_capacity"] == 64
    originals = [r for r in rows if r["operator"] == "MatMulNBitsExact"]
    assert collections.Counter(r["requests"] for r in originals) == {128: 1, 256: 8}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-accumulation", action="store_true")
    args = parser.parse_args()
    fp32 = args.fp32_accumulation
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-qkv-predecode8/build-v1"),
         prefix=ROOT / f"dense-qkv-predecode8-v1-{'fp32-' if fp32 else ''}native-gpu7-20261031",
         baseline_path=ROOT / "qmoe-prefill-rows4-v2-standard-contexts-20261031.qualification.json",
         report_status=("dense_qkv_predecode8_fp32_accumulation_fallback_finite_bitwise_qualified" if fp32 else
                        "dense_qkv_predecode8_changing_inputs_finite_bitwise_and_replay_qualified"),
         extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                     f"ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION={int(fp32)} ",
         fp32_accumulation=fp32, expect_replay=not fp32, stats_verifier=verify_stats)
