from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-pad34/build-v1"),
         prefix=ROOT / "hc-post-norm-pad34-v1-native-gpu7-20261030",
         report_status="hc_post_norm_pad34_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=ROOT / "hc-up-parallel2-v1-contexts-20261029.qualification.json")
