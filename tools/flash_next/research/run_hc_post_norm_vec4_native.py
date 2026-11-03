from pathlib import Path

from run_hc_bitcast_native import main
from qualify_quickgelu_mul_long_context import ROOT


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-vec4/build-v1"),
         prefix=ROOT / "hc-post-norm-vec4-v1-native-gpu7-20261030",
         report_status="hc_post_norm_vec4_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v2/build-manifest.json"))
