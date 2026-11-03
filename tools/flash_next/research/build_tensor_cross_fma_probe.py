import argparse
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once
from build_tensor_dot4_probe import FLAG
from build_tensor_product_probe import transform as products


OUTPUT = Path("/workspace/qwen38-int4-tensor-dot4-cross-fma-probe/build-v1")
ORIGINAL_DOT = """fn product_dot4(row: u32, col: u32) -> f16 {
  let p = vec4<f16>(products[row * 16u + col],
                   products[(row + 1u) * 16u + col],
                   products[(row + 2u) * 16u + col],
                   products[(row + 3u) * 16u + col]);
  return dot(vec4<f16>(1), p);
}"""
CROSS_DOT = """fn product_dot4(row: u32, col: u32) -> f16 {
  return products[row * 16u + col] + products[(row + 1u) * 16u + col];
}"""


def transform(source, materialize=False, fp32_stages=False):
    materialize = materialize or fp32_stages
    source = products(source)
    source = replace_once(source, '"TensorProductProbeF16"', '"TensorCrossFmaProbeF16"')
    source = replace_once(source, ORIGINAL_DOT, CROSS_DOT)
    source = replace_once(source, '", row == col);\\n"',
                          '", row == col && row % 4u < 2u);\\n"')
    source = replace_once(source,
        '''         << "result = subgroupMatrixMultiplyAccumulate(left, right, result);\\n"
         << "subgroupMatrixStore<row_major>(&products, 0u, result, 16u);\\n"''',
        '''         << "result = subgroupMatrixMultiplyAccumulate(left, right, result);\\n"
         << "workgroupBarrier();\\n"
         << "for (var i = local_idx; i < 256u; i += 32u) {\\n"
         << "  let row = i / 16u;\\n"
         << "  let col = i % 16u;\\n"
         << "  diagonal_a[i] = select(f16(0), " << a.GetByOffset("col")
         << ", col == row + 2u && row % 4u < 2u);\\n"
         << "}\\n"
         << "workgroupBarrier();\\n"
         << "let shifted_left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>, row_major>"
            "(&diagonal_a, 0u, 16u);\\n"
         << "result = subgroupMatrixMultiplyAccumulate(shifted_left, right, result);\\n"
         << "subgroupMatrixStore<row_major>(&products, 0u, result, 16u);\\n"''')
    if materialize:
        source = replace_once(source,
            '''         << "result = subgroupMatrixMultiplyAccumulate(left, right, result);\\n"
         << "workgroupBarrier();\\n"''',
            '''         << "result = subgroupMatrixMultiplyAccumulate(left, right, result);\\n"
         << "subgroupMatrixStore<row_major>(&products, 0u, result, 16u);\\n"
         << "workgroupBarrier();\\n"''')
        source = replace_once(source,
            '''         << "result = subgroupMatrixMultiplyAccumulate(shifted_left, right, result);\\n"''',
            '''         << "let rounded_result = subgroupMatrixLoad<subgroup_matrix_result<f16, 16, 16>, row_major>"
            "(&products, 0u, 16u);\\n"
         << "result = subgroupMatrixMultiplyAccumulate(shifted_left, right, rounded_result);\\n"''')
    if fp32_stages:
        source = replace_once(source, '"TensorCrossFmaProbeF16"', '"TensorCrossFmaF32Probe"')
        source = replace_once(source, "{{f16, f16, 16, 16, 16, 32, true}}",
                              "{{f16, wgpu::SubgroupMatrixComponentType::F32, 16, 16, 16, 32, true}}")
        source = replace_once(source, "var<workgroup> products: array<f16, 256>;",
                              "var<workgroup> products: array<f16, 256>;\n"
                              "var<workgroup> scratch_f32: array<f32, 256>;")
        assert source.count("subgroup_matrix_result<f16, 16, 16>") == 2
        source = source.replace("subgroup_matrix_result<f16, 16, 16>",
                                "subgroup_matrix_result<f32, 16, 16>")
        assert source.count("subgroupMatrixStore<row_major>(&products") == 2
        source = source.replace("subgroupMatrixStore<row_major>(&products",
                                "subgroupMatrixStore<row_major>(&scratch_f32")
        source = replace_once(source, '"(&products, 0u, 16u);\\n"',
                              '"(&scratch_f32, 0u, 16u);\\n"')
        source = replace_once(source,
            '''         << ", col == row + 2u && row % 4u < 2u);\\n"
         << "}\\n"''',
            '''         << ", col == row + 2u && row % 4u < 2u);\\n"
         << "  scratch_f32[i] = unpack2x16float(pack2x16float(vec2<f32>(scratch_f32[i], 0))).x;\\n"
         << "}\\n"''')
        source = replace_once(source,
            '''         << "result = subgroupMatrixMultiplyAccumulate(shifted_left, right, rounded_result);\\n"
         << "subgroupMatrixStore<row_major>(&scratch_f32, 0u, result, 16u);\\n"
         << "workgroupBarrier();\\n"''',
            '''         << "result = subgroupMatrixMultiplyAccumulate(shifted_left, right, rounded_result);\\n"
         << "subgroupMatrixStore<row_major>(&scratch_f32, 0u, result, 16u);\\n"
         << "workgroupBarrier();\\n"
         << "for (var i = local_idx; i < 256u; i += 32u) { products[i] = f16(scratch_f32[i]); }\\n"
         << "workgroupBarrier();\\n"''')
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--fp32-stages", action="store_true")
    args = parser.parse_args()
    build(output=OUTPUT.parent / ("build-v3" if args.fp32_stages else "build-v2")
          if args.materialize or args.fp32_stages else OUTPUT, flag=FLAG,
          source_transform=lambda source: transform(source, materialize=args.materialize,
                                                     fp32_stages=args.fp32_stages),
          component_dir=Path("/workspace/qwen38-int4-dense-block32/build-v1"),
          source_name="matmul_nbits.cc", comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: source,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          preserves_reduction=False)
