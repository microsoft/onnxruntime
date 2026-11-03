from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-wide16/build-v1"),
         prefix=ROOT / "hc-post-norm-wide16-v1-native-gpu7-20261031",
         report_status="hc_post_norm_wide16_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=ROOT / "dense-block32-v2-contexts-20261030.qualification.json")
