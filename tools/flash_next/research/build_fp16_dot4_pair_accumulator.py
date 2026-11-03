import json
from pathlib import Path

from build_fp16_dot4_pair_components import PROGRAM as OLD_PROGRAM, FLAG as OLD_FLAG, CLASS as OLD_CLASS, SOURCE, transform as old_transform
from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_FP16_DOT4_PAIR_ACCUMULATOR"
CLASS = "Fp16Dot4PairAccumulatorProgram"
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-pair-accumulator/build-v1")
PROGRAM = OLD_PROGRAM.replace("Fp16Dot4PairComponents", "Fp16Dot4PairAccumulator")
PROGRAM = replace_once(PROGRAM, "var<workgroup> rhs: array<f16, 256>;",
                        "var<workgroup> rhs: array<f16, 256>;\nvar<workgroup> matrix_weights: array<f16, 256>;")
PROGRAM = replace_once(PROGRAM, "if (row == 0u && k == 0u) { value = f16(1); }",
                        "if (row == 0u && k == 0u) { value = xv.w; }")
PROGRAM = replace_once(PROGRAM, "  if (row == 0u && k == 1u) { value = xv.w; }\n", "")
PROGRAM = replace_once(PROGRAM, "if (row == 1u && k == 2u) { value = f16(1); }",
                        "if (row == 1u && k == 2u) { value = xv.z; }")
PROGRAM = replace_once(PROGRAM, "  if (row == 1u && k == 3u) { value = xv.z; }\n", "")
PROGRAM = replace_once(PROGRAM, "workgroupBarrier();\nlet left = subgroupMatrixLoad", """for (var i = local_idx; i < 256u; i += 32u) {
  let k = i / 16u; let col = i % 16u; let wv = weights[col];
  var value = f16(0);
  if (k == 0u) { value = wv.w; }
  if (k == 2u) { value = wv.z; }
  matrix_weights[i] = value;
}
workgroupBarrier();
for (var i = local_idx; i < kRows * 16u; i += 32u) {
  let row = i / 16u; let col = i % 16u;
  var bias = f32(0);
  if (row == 0u) { bias = f32(rhs[col]); }
  if (row == 1u) { bias = f32(rhs[32u + col]); }
  result[i] = bias;
}
workgroupBarrier();
let left = subgroupMatrixLoad""")
PROGRAM = replace_once(PROGRAM, "row_major>(&rhs, 0u, 16u);",
                        "row_major>(&matrix_weights, 0u, 16u);")
PROGRAM = replace_once(PROGRAM, "var accumulator: subgroup_matrix_result<f32, 16, kRows>;",
                        "var accumulator = subgroupMatrixLoad<subgroup_matrix_result<f32, 16, kRows>, row_major>(&result, 0u, 16u);")


def transform(source):
    result = replace_once(old_transform(source), OLD_PROGRAM, PROGRAM)
    result = result.replace(OLD_FLAG, FLAG)
    result = replace_once(result, OLD_CLASS + " program", CLASS + " program")
    return replace_once(result, """        {Type::F16, Type::F32, 16, 16, 16, 32, false},
        {Type::F16, Type::F32, 8, 16, 16, 32, false}""",
                        "        {Type::F16, Type::F32, 16, 16, 16, 32, false}")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, matrix_rows=16,
                        seeded_fp32_accumulator_from_materialized_fp16_products=True,
                        cross_shader_comparator_required=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
