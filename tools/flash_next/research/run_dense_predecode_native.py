import argparse
import collections
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main


def verify_predecoded_stats(rows):
    assert collections.Counter(row["operator"] for row in rows) == {
        "MatMulNBitsExact": 9, "DensePredecoded": 9}
    for row in rows:
        assert row["requests"] == 128
        assert row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0
        assert row["hits"] + row["captures"] == row["requests"]
        assert row["entries"] == 1 and row["capacity"] == 64
        assert 0 < row["binding_entries"] <= row["binding_capacity"] == 64


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-accumulation", action="store_true")
    args = parser.parse_args()
    fp32 = args.fp32_accumulation
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-predecode/build-v2"),
         prefix=ROOT / f"dense-predecode-v2-{'fp32-' if fp32 else ''}native-gpu7-20261031",
         baseline_path=ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json",
         report_status=("dense_predecode_fp32_accumulation_fallback_finite_bitwise_qualified" if fp32 else
                        "dense_predecode_changing_inputs_finite_bitwise_and_replay_qualified"),
         extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                     f"ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION={int(fp32)} ",
         fp32_accumulation=fp32, expect_replay=not fp32,
         stats_verifier=verify_predecoded_stats)
