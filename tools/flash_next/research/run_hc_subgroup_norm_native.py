import argparse
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    main(directory=Path(f"/home/sayanshaw/repos/qwen38-int4-hc-subgroup-norm/build-v{args.version}"),
         prefix=ROOT / f"hc-subgroup-norm-v{args.version}-native-gpu7-20261029",
         report_status="hc_subgroup_four_outputs_finite_bitwise_and_replay_qualified")
