from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_fc2_four_columns_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-fc2-compact20/build-v1"),
         prefix=ROOT / "fc2-compact20-v1-native-gpu7-20261031",
         baseline_name="qmoe-prefill-rows4-v2-mode1-gpu7-full-profile2048-20261031",
         report_status="fc2_compact20_changing_inputs_routes_finite_bitwise_and_replay_qualified",
         variant_field="compact_lanes_tested", variant_values=(0, 1))
