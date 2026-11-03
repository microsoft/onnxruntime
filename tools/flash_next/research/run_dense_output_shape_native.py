import json
from pathlib import Path

from prepare_generic_matmul_replay import SHAPES
from qualify_quickgelu_mul_long_context import ROOT
from run_dense_bitcast_native import main as run_native


PREFIX = ROOT / "dense-output-shape-v2-native-gpu7-20261031"
SHAPE_STATS = ROOT / (PREFIX.name + "-shape-stats.jsonl")


def verify_stats(rows):
    assert len(rows) == 9
    assert all(row["operator"] == "MatMulNBitsOutputShapeExact" and row["requests"] == 256 and
               row["hits"] == 255 and row["captures"] == 1 and row["entries"] == 1 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and row["binding_hits"] > 0
               for row in rows), rows
    shapes = [json.loads(line) for line in SHAPE_STATS.read_text().splitlines()]
    assert len(shapes) == 27, len(shapes)
    fast = [row for row in shapes if row["fast_shape_hits"] > 0]
    assert len(fast) == 9 and {(row["K"], row["N"]) for row in fast} == set(SHAPES)
    assert all(row["fast_shape_hits"] == 128 for row in fast), fast


def main():
    if SHAPE_STATS.exists():
        raise FileExistsError(SHAPE_STATS)
    run_native(directory=Path("/home/sayanshaw/repos/qwen38-int4-dense-output-shape/build-v2"),
        prefix=PREFIX,
        baseline_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
        report_status="dense_output_shape_changing_inputs_finite_bitwise_qualified",
        extra_flags="ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
                    "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 "
                    "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE="
                    f"/evidence/webgpu-perf-oct20/{SHAPE_STATS.name} ",
        fp32_accumulation=False, stats_verifier=verify_stats)


if __name__ == "__main__":
    main()
