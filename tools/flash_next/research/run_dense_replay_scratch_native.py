import argparse
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main as run_native


def verify_stats(rows):
    assert len(rows) == 9, len(rows)
    assert all(row["operator"] == "MatMulNBitsScratchExact" and
               row["requests"] == 256 and row["hits"] == 255 and row["captures"] == 1 and
               row["entries"] == 1 and row["rebinds"] + row["binding_hits"] == row["hits"] and
               row["binding_hits"] > 0 and row["capacity"] == 64 for row in rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    run_native(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-replay-scratch/build-v1"),
        prefix=ROOT / "dense-replay-scratch-v1-native-gpu7-20261031",
        baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
        report_status="dense_replay_scratch_changing_inputs_finite_bitwise_qualified",
        extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                    "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
        fp32_accumulation=False, verify_existing=args.verify_existing, stats_verifier=verify_stats)


if __name__ == "__main__":
    main()
