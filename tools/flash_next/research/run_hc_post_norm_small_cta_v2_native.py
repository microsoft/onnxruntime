from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main


if __name__ == "__main__":
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                    "--format=csv"], check=True)
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-small-cta/build-v2"),
         prefix=ROOT / "hc-post-norm-small-cta-v2-native-gpu7",
         report_status="hc_post_norm_small_cta_four_outputs_finite_bitwise_and_replay_qualified",
         parent_path=Path("/home/sayanshaw/repos/qwen38-int4-fc2-private-a/build-v1/build-manifest.json"))
