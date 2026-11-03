from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-tile4/build-v1"),
         prefix=ROOT / "dense-tile4-v1-native-gpu7-20261030",
         baseline_path=ROOT / "hc-up-parallel2-v1-contexts-20261029.qualification.json",
         report_status="dense_tile4_changing_inputs_finite_bitwise_and_replay_qualified",
         extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ")
