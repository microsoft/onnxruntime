import json
from pathlib import Path
import subprocess

from build_fp16_dot4_pair_matrix_probe import FLAG as OLD_FLAG, PROGRAM as OLD_PROGRAM, SOURCE, transform as old_transform
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_PAIR_MATRIX_MATERIALIZED"
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-pair-matrix-materialized/build-v1")
CLASS = "Fp16Dot4PairMatrixMaterializedProgram"
PROGRAM = OLD_PROGRAM.replace("Fp16Dot4PairMatrixProbe", "Fp16Dot4PairMatrixMaterialized")
PROGRAM = replace_once(PROGRAM, "var<workgroup> result: array<f32, 256>;", """var<workgroup> result: array<f32, 256>;
var<workgroup> native_pairs: array<vec2<f16>, 16>;
var<workgroup> guarded_pairs: array<vec2<f16>, 16>;
var<workgroup> masks: array<u32, 16>;""")
PROGRAM = replace_once(PROGRAM, "select(0u, 16u, bound0) | select(0u, 32u, bound1);\n)", """select(0u, 16u, bound0) | select(0u, 32u, bound1);
  native_pairs[col] = vec2<f16>(native0, native1);
  guarded_pairs[col] = vec2<f16>(pair0, pair1);
  masks[col] = mask;
}
workgroupBarrier();
if (local_idx < 16u) {
  let col = local_idx; let wv = weights[col];
  let rounded_native = native_pairs[col];
  let rounded_guarded = guarded_pairs[col];
  let mask = masks[col];
)""")
PROGRAM = replace_once(PROGRAM, '"native0 + native1"', '"rounded_native.x + rounded_native.y"')
PROGRAM = replace_once(PROGRAM, '"pair0 + pair1"', '"rounded_guarded.x + rounded_guarded.y"')


def transform(source):
    result = replace_once(old_transform(source), OLD_PROGRAM, PROGRAM)
    result = result.replace(OLD_FLAG, FLAG)
    return replace_once(result, "Fp16Dot4PairMatrixProbeProgram program", CLASS + " program")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, matrix_result_type="f32",
                        pair_tree="fma3_rounded0_plus_fma2_rounded1",
                        scalar_zero_subnormal_boundary_correction=True,
                        fp16_pair_materialization=True))
    runner = OUTPUT / "pair_matrix_compare.cpp"
    runner.write_text((ROOT / "fp16_dot4_pair_matrix_compare.cpp").read_text())
    subprocess.run(["c++", "-O3", "-std=c++20", str(runner),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "pair_matrix_compare")], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  diagnostic_runner_sha256=sha(OUTPUT / "pair_matrix_compare"))
    report["source_sha256"]["pair_matrix_compare.cpp"] = sha(runner)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
