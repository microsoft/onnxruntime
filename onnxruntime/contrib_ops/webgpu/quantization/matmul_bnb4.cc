// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_bnb4.h"

#include <string>

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status MatMulBnb4Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::None);
  const auto& absmax = shader.AddInput("absmax", ShaderUsage::UseValueTypeAlias);
  const auto& y = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // `a_components` (1/2/4) is A's K vectorization factor and `output_number` (1/2/4/8) is the
  // number of output rows each invocation computes; both drive constant-bounded loops in the
  // template (the WGSL compiler unrolls them). The FP4/NF4 dequantization table is selected by
  // `quant_type`.
  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_bnb4.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(a_components, a.NumComponents()),
                             WGSL_TEMPLATE_PARAMETER(output_number, output_number_),
                             WGSL_TEMPLATE_PARAMETER(quant_type, quant_type_),
                             WGSL_TEMPLATE_VARIABLE(absmax, absmax),
                             WGSL_TEMPLATE_VARIABLE(input_a, a),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output, y));
}

Status MatMulBnb4TileProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::None);
  const auto& absmax = shader.AddInput("absmax", ShaderUsage::UseValueTypeAlias);
  const auto& y = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  // `components == 4` selects the vec4 GEMM tiling (N % 4 == 0 && K % 4 == 0); otherwise the scalar
  // 16x16 tile is used. `tile` is the per-branch tile edge (kGemmTile for vec4, kTileSize for
  // scalar) and `rpt` (kGemmRowsPerThread) is only consumed by the vec4 branch. The shared-tile
  // storage type follows the input element type (input_a_element_t) to halve workgroup memory for
  // f16 inputs; the accumulator stays f32 for precision.
  const int tile = components_ == 4 ? MatMulBnb4TileProgram::kGemmTile : MatMulBnb4TileProgram::kTileSize;
  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_bnb4_tile.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components_),
                             WGSL_TEMPLATE_PARAMETER(quant_type, quant_type_),
                             WGSL_TEMPLATE_PARAMETER(rpt, MatMulBnb4TileProgram::kGemmRowsPerThread),
                             WGSL_TEMPLATE_PARAMETER(tile, tile),
                             WGSL_TEMPLATE_VARIABLE(absmax, absmax),
                             WGSL_TEMPLATE_VARIABLE(input_a, a),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output, y));
}

Status MatMulBnb4::ComputeInternal(ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);
  const Tensor* absmax = context.Input(2);

  const auto& a_shape = a->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "Input A must have at least 1 dimension.");
  ORT_RETURN_IF_NOT(a_shape[a_shape.NumDimensions() - 1] == K_,
                    "The last dimension of input A (", a_shape[a_shape.NumDimensions() - 1],
                    ") must match K (", K_, ").");

  // Overflow-safe expected sizes (K_, N_, block_size_ validated > 0 in the constructor).
  ORT_RETURN_IF_NOT(K_ <= std::numeric_limits<int64_t>::max() / N_,
                    "Overflow computing K * N for K=", K_, ", N=", N_, ".");
  const int64_t numel = K_ * N_;
  const int64_t expected_b_size = ((numel - 1) / 2) + 1;
  const int64_t expected_absmax_size = ((numel - 1) / block_size_) + 1;
  ORT_RETURN_IF_NOT(b->Shape().Size() >= expected_b_size,
                    "B tensor size (", b->Shape().Size(), ") is too small for K=", K_,
                    " and N=", N_, ". Expected at least ", expected_b_size, " elements.");
  ORT_RETURN_IF_NOT(absmax->Shape().Size() >= expected_absmax_size,
                    "absmax tensor size (", absmax->Shape().Size(), ") is too small for K=", K_,
                    ", N=", N_, ", block_size=", block_size_, ". Expected at least ",
                    expected_absmax_size, " elements.");

  // Output shape is the input A shape with the last dimension replaced by N.
  TensorShapeVector output_dims = a_shape.AsShapeVector();
  output_dims.back() = N_;
  TensorShape output_shape(output_dims);
  Tensor* y = context.Output(0, output_shape);
  const int64_t output_size = output_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }

  // Number of logical rows in A (all leading dims folded into M).
  const int64_t m = output_size / N_;

  // For larger M the shared-memory tiled kernel amortizes both the global A loads and the weight
  // dequantization across a tile of rows/columns. For small M (e.g. decode, M == 1) the row-tiled
  // kernel below has less overhead, so fall back to it.
  constexpr int64_t kTileThreshold = MatMulBnb4TileProgram::kTileSize;
  if (m >= kTileThreshold) {
    // Use the vec4 GEMM tiling when both N and K are multiples of 4 (A is read vec4 along K, B is
    // dequantized into vec4 output columns, and the output is stored vec4); otherwise fall back to
    // the scalar 16x16 tile, which handles arbitrary N and K.
    const bool use_vec4 = (N_ % 4 == 0) && (K_ % 4 == 0);
    const int components = use_vec4 ? 4 : 1;
    MatMulBnb4TileProgram program{quant_type_, components};
    program
        .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, components})
        .AddInput({b, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4})
        .AddInput({absmax, ProgramTensorMetadataDependency::TypeAndRank})
        .AddOutput({y, ProgramTensorMetadataDependency::None, components});

    if (use_vec4) {
      // 8x8 workgroup; each workgroup computes a 32x32 (kGemmTile) output block.
      constexpr uint32_t wg = MatMulBnb4TileProgram::kGemmWorkgroup;
      constexpr int64_t gtile = MatMulBnb4TileProgram::kGemmTile;
      program.SetWorkgroupSize(wg, wg, 1)
          .SetDispatchGroupSize(static_cast<uint32_t>(CeilDiv<int64_t>(N_, gtile)),
                                static_cast<uint32_t>(CeilDiv<int64_t>(m, gtile)),
                                1);
    } else {
      constexpr uint32_t tile = MatMulBnb4TileProgram::kTileSize;
      program.SetWorkgroupSize(tile, tile, 1)
          .SetDispatchGroupSize(static_cast<uint32_t>(CeilDiv<int64_t>(N_, tile)),
                                static_cast<uint32_t>(CeilDiv<int64_t>(m, tile)),
                                1);
    }

    program
        .AddUniformVariables({{static_cast<uint32_t>(m)},
                              {static_cast<uint32_t>(N_)},
                              {static_cast<uint32_t>(K_)},
                              {static_cast<uint32_t>(block_size_)}})
        .CacheHint(std::to_string(quant_type_), std::to_string(components));

    return context.RunProgram(program);
  }

  // Vectorize the K loop (A is dense; block_size is a multiple of 4 so codes within a vector share
  // a block) and compute multiple rows per invocation to reuse each dequantized weight.
  const int a_components = GetMaxComponents(K_);
  const int output_number = GetMaxComponents(m);
  const int64_t dispatch_elements = (m / output_number) * N_;

  MatMulBnb4Program program{quant_type_, output_number};
  program
      .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, a_components})
      .AddInput({b, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4})
      .AddInput({absmax, ProgramTensorMetadataDependency::TypeAndRank})
      .AddOutput({y, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize(static_cast<uint32_t>(CeilDiv<int64_t>(dispatch_elements, WORKGROUP_SIZE)))
      .AddUniformVariables({{static_cast<uint32_t>(N_)},
                            {static_cast<uint32_t>(K_)},
                            {static_cast<uint32_t>(block_size_)},
                            {static_cast<uint32_t>(dispatch_elements)}})
      .CacheHint(std::to_string(quant_type_), std::to_string(a_components), std::to_string(output_number));

  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(
    MatMulBnb4,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulBnb4);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
