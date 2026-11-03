import argparse
from pathlib import Path

from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-accumulation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    fp32 = args.fp32_accumulation
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-double-buffer/build-v1"),
         prefix=ROOT / f"dense-double-buffer-v1-{'fp32-' if fp32 else ''}native-gpu7-20261030",
         baseline_path=ROOT / "prefill1024-v2-contexts-20261030.qualification.json",
         report_status=("dense_double_buffer_fp32_accumulation_fallback_finite_bitwise_qualified" if fp32 else
                        "dense_double_buffer_changing_inputs_finite_bitwise_and_replay_qualified"),
         extra_flags=f"ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 ORT_TEST_DENSE_FP32_ACCUMULATION={int(fp32)} ",
         fp32_accumulation=fp32, expect_replay=not fp32, verify_existing=args.verify_existing)
