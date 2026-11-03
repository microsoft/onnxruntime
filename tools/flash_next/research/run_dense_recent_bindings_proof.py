from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main as run_native


def verify_bindings(rows):
    assert len(rows) == 9
    assert all(row["operator"] == "MatMulNBitsRecentExact" and row["requests"] == 256 and
               row["hits"] == 255 and row["captures"] == 1 and row["entries"] == 1 and
               row["rebinds"] >= 2 and row["binding_entries"] >= 3 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and
               row["recent_binding_hits"] == 0 and row["binding_hits"] > 0 for row in rows), rows
    log = (ROOT / "dense-recent-binding-v1-bank3-native-gpu7-20261031.log").read_text()
    assert log.count("Binding proof distinct_live_gpu_inputs=3\n") == 27


def main():
    run_native(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-recent-binding/binding-proof-v1"),
        prefix=ROOT / "dense-recent-binding-v1-bank3-native-gpu7-20261031",
        baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
        report_status="dense_recent_binding_distinct_bank_miss_path_finite_bitwise_qualified",
        extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                    "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
        fp32_accumulation=False, stats_verifier=verify_bindings)


if __name__ == "__main__":
    main()
