// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul.h"
#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/data_transfer.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MatMul);

static std::string CalcResult(int64_t components, int64_t a_components, int64_t output_number) {
  std::ostringstream oss;
  oss << "var a_data: a_value_t;\n";
  for (int i = 0; i < a_components; ++i) {
    oss << "let b_data" << i << " = b[(b_offset + (k + " << i << ") * uniforms.N + col) / " << components << "];\n";
  }
  for (int i = 0; i < output_number; ++i) {
    oss << "a_data = a[(a_offset + (row + " << i << ") * uniforms.K + k) / " << a_components << "];\n";

    for (int j = 0; j < a_components; j++) {
      oss << "values[" << i << "] = fma(b_value_t(a_data" << (a_components == 1 ? "" : "[" + std::to_string(j) + "]") << "), b_data" << j << ", values[" << i << "]);\n";
    }
  }
  return oss.str();
}

Status MatMulNaiveProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias |
                                           ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  std::string process_bias;
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
    process_bias = is_channels_last_ ? "value += output_value_t(bias[col]);" : "value += output_value_t(bias[row + i]);";
  }

  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                      ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& batch_dims = shader.AddIndices("batch_dims");

  int a_components = a.NumComponents();
  int components = b.NumComponents();  // components of N

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let col = (global_idx % (uniforms.N / " << components << ")) * " << components << ";\n"
                            << "var index1 = global_idx / (uniforms.N / " << components << ");\n"
                            << "let stride1 = uniforms.M / " << output_number_ << ";\n"
                            << "let row = (index1 % stride1) * " << output_number_ << ";\n"
                            << "let batch = index1 / stride1;\n";
  if (output_rank_ != 2) {
    shader.MainFunctionBody() << "let batch_indices = " << batch_dims.OffsetToIndices("batch") << ";\n";
  }
  shader.MainFunctionBody() << "var a_indices: a_indices_t;\n"
                            << ConvertOutputBatchIndicesToInputBatchIndices("a", a, a.Rank() - 2, batch_dims.Rank(), "batch_indices")
                            << a.IndicesSet("a_indices", a.Rank() - 2, 0) << "\n"
                            << a.IndicesSet("a_indices", a.Rank() - 1, 0) << "\n"
                            << "let a_offset = " << a.IndicesToOffset("a_indices") << "*" << a_components << ";\n"
                            << "var b_indices: b_indices_t;\n"
                            << ConvertOutputBatchIndicesToInputBatchIndices("b", b, b.Rank() - 2, batch_dims.Rank(), "batch_indices")
                            << b.IndicesSet("b_indices", b.Rank() - 2, 0) << "\n"
                            << b.IndicesSet("b_indices", b.Rank() - 1, 0) << "\n"
                            << "let b_offset = " << b.IndicesToOffset("b_indices") << " * " << components << ";\n"
                            << "var values: array<output_value_t, " << output_number_ << ">;\n"
                            << "for (var k: u32 = 0u; k < uniforms.K; k = k + " << a_components << ") {\n"
                            << CalcResult(components, a_components, output_number_) << "\n"
                            << "}\n"
                            << "for (var i = 0u; i < " << output_number_ << "u; i++) {\n"
                            << "  var value = values[i];\n"
                            << process_bias << "\n"
                            << apply_activation << "\n"
                            << "  let cur_indices = output_indices_t(batch, row + i, col/ " << components << ");\n"
                            << "  let offset = " << output.IndicesToOffset("cur_indices") << ";\n"
                            << output.SetByOffset("offset", "value")
                            << "}\n";

  return Status::OK();
}

Status MatMulTiledSubgroupProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "math/matmul_tiled_subgroup.wgsl.template");
}

Status MatMul::ComputeInternal(ComputeContext& context) const {
  // calculate output shape
  MatMulComputeHelper helper;
  const auto* a = context.Input(0);
  const auto* b = context.Input(1);

  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  auto* output_tensor = context.Output(0, helper.OutputShape());
  if (output_tensor->Shape().Size() == 0) {
    // If the output tensor is empty, we can return early.
    return Status::OK();
  }
  bool has_bias = context.InputCount() > 2;

  if (helper.N() < 8 && helper.K() < 8) {  // call MatMulNaiveProgram

    const uint32_t m = narrow<uint32_t>(helper.M());  // left matrix first dimension
    const uint32_t n = narrow<uint32_t>(helper.N());  // right matrix second dimension
    const uint32_t k = narrow<uint32_t>(helper.K());  // right matrix first dimension

    const auto components = GetMaxComponents(n);
    const auto a_components = GetMaxComponents(k);

    const auto output_number = GetMaxComponents(m);
    uint32_t output_size = narrow<uint32_t>(helper.OutputShape().Size() / components / output_number);

    const size_t output_rank = helper.OutputShape().NumDimensions();
    TensorShape outer_dims = output_rank > 2 ? helper.OutputShape().Slice(0, output_rank - 2) : TensorShape({});
    const int64_t batch_size = outer_dims.Size();

    const int64_t a_rows = a->Shape().NumDimensions() > 1 ? a->Shape()[a->Shape().NumDimensions() - 2] : 1;
    TensorShape output_shape_shader({batch_size, a_rows, helper.N() / components});

    MatMulNaiveProgram program{Activation(), output_rank, output_number, has_bias};

    program
        .CacheHint(std::to_string(components), std::to_string(a_components), std::to_string(output_number))
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_components},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, components}});

    if (has_bias) {
      const auto* bias = context.Input(2);
      program.AddInput({bias, ProgramTensorMetadataDependency::Rank, 1});
    }
    program
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::None, output_shape_shader, components}})
        .SetDispatchGroupSize((output_size + 63) / 64)  // Integer ceiling division
        .AddIndices(outer_dims)
        .AddUniformVariables({{output_size}, {m}, {n}, {k}});

    return context.RunProgram(program);
  }

  {
    auto a_shape = a->Shape();
    auto b_shape = b->Shape();
    bool are_matrices = a_shape.NumDimensions() >= 2 && b_shape.NumDimensions() >= 2;

    TensorShape batch_dims_a = a_shape.NumDimensions() > 2
                                   ? a_shape.Slice(0, a_shape.NumDimensions() - 2)
                                   : TensorShape({});
    TensorShape batch_dims_b = b_shape.NumDimensions() > 2
                                   ? b_shape.Slice(0, b_shape.NumDimensions() - 2)
                                   : TensorShape({});

    const bool is_vec4 = helper.K() % 4 == 0 && helper.N() % 4 == 0;

    // TODO: Implement broadcasting for batch dimensions.
    if (!has_bias && are_matrices && batch_dims_a == batch_dims_b && is_vec4) {
      const uint32_t m = narrow<uint32_t>(helper.M());
      const uint32_t n = narrow<uint32_t>(helper.N());
      const uint32_t k = narrow<uint32_t>(helper.K());

      auto output_shape = output_tensor->Shape();

      TensorShape batch_dims = output_shape.NumDimensions() > 2
                                   ? output_shape.Slice(0, output_shape.NumDimensions() - 2)
                                   : TensorShape({});
      const uint32_t batch_size = narrow<uint32_t>(batch_dims.Size());

      const uint32_t kTileM = 64;
      const uint32_t kTileN = 64;
      const uint32_t kMTiles = (m + kTileM - 1) / kTileM;
      const uint32_t kNTiles = (n + kTileN - 1) / kTileN;

      MatMulTiledSubgroupProgram program;
      program.SetWorkgroupSize(64, 1, 1);
      program.SetDispatchGroupSize(kNTiles, kMTiles, batch_size);

      program.AddInput({a,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        4});
      program.AddInput({b,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        4});
      program.AddOutput({output_tensor,
                         ProgramTensorMetadataDependency::TypeAndRank,
                         4});
      program.AddUniformVariables({{narrow<uint32_t>(batch_size)},
                                   {m},
                                   {k},
                                   {k / 4},
                                   {n / 4},
                                   {kMTiles},
                                   {kNTiles}});

      return context.RunProgram(program);
    }
  }

  std::vector<const Tensor*> inputs(has_bias ? 3 : 2);
  inputs[0] = a;
  inputs[1] = b;
  if (has_bias) {
    const auto* bias = context.Input(2);
    inputs.push_back(bias);
  }
  auto program = CreateMatMulProgram(Activation(), inputs, output_tensor, false);

  return context.RunProgram(program);
}

MatMulProgram CreateMatMulProgram(const Activation& activation, std::vector<const Tensor*>& inputs, Tensor* output_tensor, bool is_channels_last,
                                  const TensorShape& input_a_reshape,
                                  const TensorShape& input_b_reshape) {
  const auto* a = inputs[0];
  const auto* b = inputs[1];
  bool has_bias = inputs.size() > 2;
  TensorShape a_shape = input_a_reshape.NumDimensions() > 0 ? input_a_reshape : a->Shape();
  TensorShape b_shape = input_b_reshape.NumDimensions() > 0 ? input_b_reshape : b->Shape();

  MatMulComputeHelper helper;
  ORT_THROW_IF_ERROR(helper.Compute(a_shape, b_shape));
  int64_t batchA = a_shape.SizeToDimension(a_shape.NumDimensions() - 2);
  int64_t batchB = b_shape.SizeToDimension(b_shape.NumDimensions() - 2);

  TensorShape output_shape = helper.OutputShape();

  const int64_t dim_output_outer = output_shape[output_shape.NumDimensions() - 2];
  // check if A is  batch of vector (bach is not 1, M is 1) and B is a matrix (batch is 1)
  if (batchA != 1 && dim_output_outer == 1 && batchB == 1) {
    // optimization for batched vector matrix multiplication
    // dimensions of A: [1,`batchA`,K]
    TensorShapeVector dims_a = {1, batchA, helper.K()};
    // dimensions of B: [1,K,N]
    TensorShapeVector dims_b = {1, helper.K(), helper.N()};

    a_shape = TensorShape(dims_a);
    b_shape = TensorShape(dims_b);
    output_shape = {1, batchA, helper.N()};
  }

  // helpful dimension variables
  TensorShape outer_dims_a = a_shape.NumDimensions() > 2
                                 ? a_shape.Slice(0, a_shape.NumDimensions() - 2)
                                 : TensorShape({});

  TensorShape outer_dims_b = b_shape.NumDimensions() > 2
                                 ? b_shape.Slice(0, b_shape.NumDimensions() - 2)
                                 : TensorShape({});

  TensorShape outer_dims = output_shape.NumDimensions() > 2
                               ? output_shape.Slice(0, output_shape.NumDimensions() - 2)
                               : TensorShape({});

  const int64_t batch_size = outer_dims.Size();

  // Get dimensions for matrix multiplication from TensorShape
  const uint32_t dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);  // left matrix second dimension
  const uint32_t dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);    // left matrix first dimension
  const uint32_t dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 1]);  // right matrix first dimension

  const bool is_vec4 = dim_inner % 4 == 0 && dim_b_outer % 4 == 0;

  InlinedVector<int64_t> elements_per_thread = dim_a_outer <= 8
                                                   ? InlinedVector<int64_t>({4, 1, 1})
                                                   : InlinedVector<int64_t>({4, 4, 1});

  const uint32_t dispatch_x = narrow<uint32_t>((dim_b_outer + MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X * elements_per_thread[0] - 1) /
                                               (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X * elements_per_thread[0]));
  const uint32_t dispatch_y = narrow<uint32_t>((dim_a_outer + MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y * elements_per_thread[1] - 1) /
                                               (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y * elements_per_thread[1]));
  const uint32_t dispatch_z = narrow<uint32_t>((static_cast<uint32_t>(batch_size) + MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z * elements_per_thread[2] - 1) /
                                               (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z * elements_per_thread[2]));

  const int components = is_vec4 ? 4 : 1;
  const TensorShape a_shape_temp = CreateMatMulIntermediateShape(outer_dims_a, dim_a_outer, dim_inner, components);
  const TensorShape b_shape_temp = CreateMatMulIntermediateShape(outer_dims_b, dim_inner, dim_b_outer, components);
  const TensorShape output_shape_temp = TensorShape({batch_size, dim_a_outer, dim_b_outer / components});

  MatMulProgram program{activation, has_bias, is_vec4, elements_per_thread, is_channels_last};
  program
      .CacheHint(activation.ToString(), absl::StrJoin(elements_per_thread, "-"), std::to_string(is_vec4), components, is_channels_last)
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_shape_temp, components},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, b_shape_temp, components}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::Rank, output_shape_temp, components}})
      .AddUniformVariables({{dim_a_outer}, {dim_b_outer}, {dim_inner}})
      .AddIndices(outer_dims)
      .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z)
      .SetWorkgroupSize(MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X, MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y, MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z);

  if (has_bias) {
    auto bias_components = is_channels_last ? components : 1;
    const auto* bias = inputs[2];
    TensorShape reduced_bias_shape = ReduceShapeByComponents(bias->Shape(), bias_components);
    program.AddInput({bias, ProgramTensorMetadataDependency::Rank, reduced_bias_shape, bias_components});
  }
  return program;
}

}  // namespace webgpu
}  // namespace onnxruntime
