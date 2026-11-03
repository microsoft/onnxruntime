from pathlib import Path

from build_qmoe_prefill2048 import FALLBACK_ROWS, FLAG, ROWS
from qualify_quickgelu_mul_long_context import ROOT
from run_qmoe_prefill256_native import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill2048/build-v2"),
         prefix=ROOT / "qmoe-prefill2048-v2-native-gpu7-20261030",
         parent_path=ROOT / "dense-block32-v2-contexts-20261030.qualification.json",
         selector=FLAG, rows=ROWS, fallback_rows=FALLBACK_ROWS,
         expected_shapes={16: 2000, 128: 8, 256: 8, 512: 8, 1024: 8, 2048: 8},
         report_status="prefill2048_retained_chunk16_finite_bitwise_and_dispatch_qualified",
         extra_flags="ORT_WEBGPU_QMOE_PREFILL256=1 ORT_WEBGPU_QMOE_PREFILL512=1 "
                     "ORT_WEBGPU_QMOE_PREFILL1024=1 ORT_WEBGPU_DSA_QUERY_CACHE=1 "
                     "ORT_WEBGPU_DENSE_BLOCK32=1")
