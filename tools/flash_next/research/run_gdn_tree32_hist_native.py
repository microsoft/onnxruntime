from pathlib import Path

from run_gdn_value2_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-gdn-tree32-hist/build-v1"),
         prefix_name="gdn-tree32-hist-v1-native-gpu7-20261031",
         parent_variant="qsa-hist-rank-v2", parent_stamp="20261031",
         program_name="GatedDeltaNetTree32", toggle_flag="ORT_TEST_TOGGLE_GDN_TREE32",
         report_status="gdn_tree32_both_outputs_finite_bitwise_and_dispatch_qualified")
