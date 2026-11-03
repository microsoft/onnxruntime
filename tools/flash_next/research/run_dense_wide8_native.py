from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-wide8/build-v1"),
         prefix=ROOT / "dense-wide8-v1-native-gpu7-20261030",
         baseline_path=ROOT / "prefill512-v3-contexts-20261030.qualification.json",
         report_status="dense_wide8_changing_inputs_finite_bitwise_and_replay_qualified",
         extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 ")
