from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-projection-shuffle/build-v1"),
         prefix=ROOT / "hc-projection-shuffle-v1-native-gpu7-20261030",
         report_status="hc_projection_shuffle_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3/build-manifest.json"))
