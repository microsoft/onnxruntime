from pathlib import Path

from run_hc_up_tiled_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-up-parallel4/build-v1"),
         prefix_name="hc-up-parallel4-v1-native-gpu7-20261030",
         baseline_name="prefill512-v3-contexts-20261030.qualification.json",
         columns=4, control_columns=2,
         extra_flags="ORT_WEBGPU_HC_UP_PARALLEL2=1 ",
         status="hc_up_parallel4_changing_inputs_finite_bitwise_and_replay_qualified")
