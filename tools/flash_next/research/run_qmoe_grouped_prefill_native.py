from pathlib import Path

from build_qmoe_grouped_prefill import FLAG
from build_qmoe_prefill1024 import FALLBACK_ROWS, ROWS
from qualify_quickgelu_mul_long_context import ROOT
from run_qmoe_prefill256_native import main


PROGRAMS = ("QmoeGroupedExpertCount", "QmoeGroupedExpertOffsets", "QmoeGroupedExpertPermutation",
            "QmoeGroupedHiddenGather", "QmoeGroupedOutputScatter")


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1"),
         prefix=ROOT / "qmoe-grouped-prefill-v1-native-gpu7-20261031",
         parent_path=ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json",
         selector=FLAG, rows=ROWS, fallback_rows=FALLBACK_ROWS,
         expected_shapes={16: 976, 128: 8, 256: 8, 512: 8, 1024: 8},
         report_status="grouped_prefill_retained_chunk16_finite_bitwise_and_dispatch_qualified",
         extra_flags="ORT_WEBGPU_QMOE_PREFILL256=1 ORT_WEBGPU_QMOE_PREFILL512=1 ORT_WEBGPU_QMOE_PREFILL1024=1",
         additional_dispatches={name: 16 for name in PROGRAMS})
