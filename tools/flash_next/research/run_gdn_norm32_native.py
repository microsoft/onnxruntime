from pathlib import Path

from run_gdn_value2_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-gdn-norm32/build-v1"),
         prefix_name="gdn-norm32-v1-native-gpu7-20261031-attempt2", parent_variant="qsa-prefill-score16-v1",
         parent_stamp="20261031", program_name="GatedDeltaNetNorm32",
         toggle_flag="ORT_TEST_TOGGLE_GDN_NORM32", norm_only=True,
         report_status="gdn_norm32_both_outputs_finite_bitwise_and_dispatch_qualified")
