from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main as run_native
from run_dense_packed4_native import verify_packed_stats


def verify_bindings(rows):
    verify_packed_stats(rows)
    assert all(row["rebinds"] >= 2 and row["binding_entries"] >= 3 for row in rows), rows


def main():
    def verify_native(rows):
        log = (ROOT / "dense-packed4-v4-binding3-native-gpu7-20261031.log").read_text()
        assert log.count("Binding proof distinct_live_gpu_inputs=3\n") == 27
        verify_bindings(rows)

    run_native(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-packed4/binding-proof-v1"),
               prefix=ROOT / "dense-packed4-v4-binding3-native-gpu7-20261031",
               baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
               report_status="dense_packed4_distinct_live_gpu_inputs_finite_bitwise_and_rebinding_qualified",
               extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                           "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
               fp32_accumulation=False, stats_verifier=verify_native)


if __name__ == "__main__":
    main()
