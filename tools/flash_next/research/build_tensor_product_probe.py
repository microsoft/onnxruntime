from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once
from build_tensor_dot4_probe import FLAG, transform as transform_dot4


PROGRAM = r'''
class TensorProductProbeProgram final : public Program<TensorProductProbeProgram> {
 public:
  TensorProductProbeProgram() : Program{"TensorProductProbeF16"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> diagonal_a: array<f16, 256>;
var<workgroup> tile_b: array<f16, 256>;
var<workgroup> products: array<f16, 256>;
fn product_dot4(row: u32, col: u32) -> f16 {
  let p = vec4<f16>(products[row * 16u + col],
                   products[(row + 1u) * 16u + col],
                   products[(row + 2u) * 16u + col],
                   products[(row + 3u) * 16u + col]);
  return dot(vec4<f16>(1), p);
}
)";
    auto& code = shader.MainFunctionBody();
    code << "for (var i = local_idx; i < 256u; i += 32u) {\n"
         << "  let row = i / 16u;\n"
         << "  let col = i % 16u;\n"
         << "  diagonal_a[i] = select(f16(0), " << a.GetByOffset("row") << ", row == col);\n"
         << "  let output_col = workgroup_idx * 16u + col;\n"
         << "  let packed = " << b.GetByOffset("output_col * 320u + row / 8u") << ";\n"
         << "  let q = i32((packed >> ((row % 8u) * 4u)) & 15u) - 8;\n"
         << "  tile_b[col * 16u + row] = f16(q) * "
         << scales.GetByOffset("output_col * 80u") << ";\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>, row_major>"
            "(&diagonal_a, 0u, 16u);\n"
         << "let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>, col_major>"
            "(&tile_b, 0u, 16u);\n"
         << "var result: subgroup_matrix_result<f16, 16, 16>;\n"
         << "result = subgroupMatrixMultiplyAccumulate(left, right, result);\n"
         << "subgroupMatrixStore<row_major>(&products, 0u, result, 16u);\n"
         << "workgroupBarrier();\n"
         << "if (local_idx < 16u) {\n"
         << "  let first = product_dot4(0u, local_idx) + product_dot4(4u, local_idx);\n"
         << "  let second = product_dot4(8u, local_idx) + product_dot4(12u, local_idx);\n"
         << "  let value = first + second;\n"
         << y.SetByOffset("workgroup_idx * 16u + local_idx", "value") << "\n"
         << "}\n";
    return Status::OK();
  }
};

'''


def transform(source):
    source = transform_dot4(source)
    source = replace_once(source, "class MatMulInt3Program final", PROGRAM + "class MatMulInt3Program final")
    return replace_once(
        source,
        '''    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, 1, 2560, 2560,
                                         4, 80, *config, context, y, 0, nullptr);''',
        '''    TensorProductProbeProgram program;
    program.SetWorkgroupSize(32).SetDispatchGroupSize(160);
    if (context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
      program.SetSubgroupSize(config->subgroupSize);
    }
    program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                       {b, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1});
    return context.RunProgram(program);''')


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-tensor-dot4-probe/build-v4"), flag=FLAG,
          source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-dense-block32/build-v1"),
          source_name="matmul_nbits.cc", comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: source,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          preserves_reduction=False)
