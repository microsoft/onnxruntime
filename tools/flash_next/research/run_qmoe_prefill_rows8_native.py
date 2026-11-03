import argparse
from pathlib import Path

from build_qmoe_prefill1024 import FALLBACK_ROWS, ROWS
from build_qmoe_prefill_rows8 import FLAG, PROGRAM
from qualify_quickgelu_mul_long_context import ROOT
from run_qmoe_grouped_prefill_native import PROGRAMS
from run_qmoe_prefill256_native import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows8/build-v1"),
         prefix=ROOT / "qmoe-prefill-rows8-v1-native-gpu7-20261031",
         parent_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
         selector=FLAG, rows=ROWS, fallback_rows=FALLBACK_ROWS,
         expected_shapes={16: 976, 128: 8, 256: 8, 512: 8, 1024: 8},
         report_status="grouped_rows8_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified",
         extra_flags="ORT_WEBGPU_QMOE_PREFILL256=1 ORT_WEBGPU_QMOE_PREFILL512=1 "
                     "ORT_WEBGPU_QMOE_PREFILL1024=1 ORT_WEBGPU_QMOE_GROUPED_PREFILL=1",
         additional_dispatches={**{name: 16 for name in PROGRAMS}, PROGRAM: 32},
         verify_existing=args.verify_existing)
