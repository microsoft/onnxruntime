from pathlib import Path

from run_gdn_value2_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-gdn-value8/build-v1"),
         prefix_name="gdn-value8-v1-native-gpu7-20261030", parent_variant="dense-block32-v2",
         program_name="GatedDeltaNetValue8", toggle_flag="ORT_TEST_TOGGLE_GDN_VALUE8",
         report_status="gdn_value8_both_outputs_finite_bitwise_and_dispatch_qualified")
