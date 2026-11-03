from pathlib import Path

from build_qmoe_prefill512 import FALLBACK_ROWS, FLAG, ROWS
from qualify_quickgelu_mul_long_context import ROOT
from run_qmoe_prefill256_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3"),
         prefix=ROOT / "qmoe-prefill512-v3-native-gpu7-20261030",
         parent_path=ROOT / "dense-tile4-v1-contexts-20261030.qualification.json",
         selector=FLAG, rows=ROWS, fallback_rows=FALLBACK_ROWS,
         expected_shapes={16: 464, 128: 8, 256: 8, 512: 8},
         report_status="prefill512_retained_chunk16_finite_bitwise_and_dispatch_qualified",
         extra_flags="ORT_WEBGPU_QMOE_PREFILL256=1")
