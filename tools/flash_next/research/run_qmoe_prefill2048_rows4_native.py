import argparse
import json
from pathlib import Path

from build_qmoe_prefill2048_rows4 import FALLBACK_ROWS, FLAG, PROGRAM, ROWS
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qmoe_grouped_prefill_native import PROGRAMS
from run_qmoe_prefill256_native import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    version = parser.parse_args().version
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-qmoe-prefill2048-rows4/build-v{version}")
    build = json.loads((directory / "build-manifest.json").read_text())
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest
    main(directory=directory,
         prefix=ROOT / f"qmoe-prefill2048-rows4-v{version}-native-gpu7-20261031",
         parent_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
         selector=FLAG, rows=ROWS, fallback_rows=FALLBACK_ROWS,
         expected_shapes={16: 2000, 128: 8, 256: 8, 512: 8, 1024: 8, 2048: 8},
         report_status="prefill2048_rows4_retained_chunk16_finite_bitwise_and_dispatch_qualified",
         extra_flags="ORT_WEBGPU_QMOE_PREFILL256=1 ORT_WEBGPU_QMOE_PREFILL512=1 "
                     "ORT_WEBGPU_QMOE_PREFILL1024=1 ORT_WEBGPU_QMOE_GROUPED_PREFILL=1",
         additional_dispatches={**{name: 24 for name in PROGRAMS},
                                "MatMulNBitsGroupedRows4F16": 32, PROGRAM: 16})
