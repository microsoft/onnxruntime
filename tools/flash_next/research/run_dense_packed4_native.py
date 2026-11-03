import argparse
import collections
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main as run_native


def verify_packed_stats(rows):
    assert collections.Counter(row["operator"] for row in rows) == {
        "MatMulNBitsExact": 9, "DensePacked4": 8}
    original_requests = sorted(row["requests"] for row in rows if row["operator"] == "MatMulNBitsExact")
    assert original_requests == [128] * 8 + [256]
    assert all(row["requests"] == 128 for row in rows if row["operator"] == "DensePacked4")
    for row in rows:
        assert row["entries"] == 1
        assert row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0
        assert row["hits"] + row["captures"] == row["requests"]
        assert row["capacity"] == 64 and 0 < row["binding_entries"] <= row["binding_capacity"] == 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-accumulation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    fp32 = args.fp32_accumulation
    run_native(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-packed4/build-v4"),
               prefix=ROOT / f"dense-packed4-v4-{'fp32-' if fp32 else ''}native-gpu7-20261031",
               baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
               report_status=("dense_packed4_fp32_accumulation_fallback_finite_bitwise_qualified" if fp32 else
                              "dense_packed4_changing_inputs_finite_bitwise_and_replay_qualified"),
               extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                           f"ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION={int(fp32)} ",
               fp32_accumulation=fp32, expect_replay=not fp32,
               verify_existing=args.verify_existing, stats_verifier=verify_packed_stats)


if __name__ == "__main__":
    main()
