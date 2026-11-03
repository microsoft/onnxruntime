import argparse
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main as run_native


def verify_stats(rows):
    assert len(rows) == 9
    assert all(row["operator"] == "MatMulNBitsExact" and row["requests"] == 257 and
               row["entries"] == 1 and row["captures"] == 1 and row["hits"] == 255 and
               row["hits"] + row["captures"] + 1 == row["requests"] and
               row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0 and
               row["capacity"] == 64 and 0 < row["binding_entries"] <= row["binding_capacity"] == 64
               for row in rows), rows
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    def verify_native(rows):
        verify_stats(rows)
        log = (ROOT / "immediate-node-replay-v1-native-gpu7-20261031.log").read_text()
        assert log.count("Experimental immediate WebGPU node replay is active.") == 1
    run_native(directory=Path("/home/sayanshaw/repos/qwen38-int4-immediate-node-replay/build-v1"),
               prefix=ROOT / "immediate-node-replay-v1-native-gpu7-20261031",
               baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
               report_status="immediate_node_replay_changing_inputs_finite_bitwise_and_order_qualified",
               extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                           "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 ",
               fp32_accumulation=False, verify_existing=args.verify_existing, stats_verifier=verify_native)


if __name__ == "__main__":
    main()
