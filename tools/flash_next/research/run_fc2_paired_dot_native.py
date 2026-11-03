from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_fc2_four_columns_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-fc2-paired-dot/build-v1"),
         prefix=ROOT / "fc2-paired-dot-v1-native-gpu7-20261031",
         baseline_name="qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031",
         report_status="fc2_paired_dot_changing_inputs_routes_finite_bitwise_and_replay_qualified",
         variant_field="paired_dot_tested", variant_values=(0, 1))
