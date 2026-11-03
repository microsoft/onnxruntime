from pathlib import Path

from run_hc_up_tiled_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-up-parallel2/build-v1"),
         prefix_name="hc-up-parallel2-v1-native-gpu7-20261029",
         baseline_name="qsa-score16-v5-contexts-20261029.qualification.json",
         columns=2, status="hc_up_parallel2_changing_inputs_finite_bitwise_and_replay_qualified")
