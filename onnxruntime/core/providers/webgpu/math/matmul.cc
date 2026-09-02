// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/matmul.h"

#include <limits>

#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/vendor/intel/math/matmul.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"

namespace onnxruntime {
namespace webgpu {

std::unique_ptr<MatMulOptImpl> CreateSubgroupMatrixMatMulImpl(const ComputeContextBase& context);

MatMulOptImpl* MatMulComputeCache::GetOrCreateSubgroupMatrixImpl(const ComputeContextBase& context) {
  std::call_once(subgroup_impl_init_flag_, [&]() {
    subgroup_impl_ = CreateSubgroupMatrixMatMulImpl(context);
  });
  return subgroup_impl_.get();
}

std::optional<MatMulNaiveProgramInfo> AnalyzeMatMulNaiveProgram(
    const TensorShape& a_shape,
    const TensorShape& b_shape,
    bool is_channels_last) {
  MatMulComputeHelper helper;
  if (!helper.Compute(a_shape, b_shape).IsOK() || helper.N() >= 8 || helper.K() >= 8) {
    return std::nullopt;
  }

  const uint32_t m = narrow<uint32_t>(helper.M());
  const uint32_t n = narrow<uint32_t>(helper.N());
  const uint32_t k = narrow<uint32_t>(helper.K());
  const uint32_t components = narrow<uint32_t>(GetMaxComponents(n));
  const uint32_t a_components = narrow<uint32_t>(GetMaxComponents(k));
  const uint32_t output_number = narrow<uint32_t>(GetMaxComponents(m));
  const TensorShape output_shape = helper.OutputShape();
  const size_t output_rank = output_shape.NumDimensions();
  const TensorShape outer_dims =
      output_rank > 2 ? output_shape.Slice(0, output_rank - 2) : TensorShape({});
  const int64_t output_rows =
      a_shape.NumDimensions() > 1 ? a_shape[a_shape.NumDimensions() - 2] : 1;

  return MatMulNaiveProgramInfo{
      m,
      n,
      k,
      components,
      a_components,
      output_number,
      narrow<uint32_t>(output_shape.Size() / components / output_number),
      is_channels_last ? components : 1u,
      output_rank,
      ReduceShapeByComponents(a_shape, a_components),
      ReduceShapeByComponents(b_shape, components),
      TensorShape({outer_dims.Size(), output_rows, n / components}),
      outer_dims};
}

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

  const int a_components = a.NumComponents();
  const int components = b.NumComponents();  // components of N

  std::string process_bias;
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
    process_bias = is_channels_last_
                       ? "value += output_value_t(bias[col / " + std::to_string(components) + "]);"
                       : "value += output_value_t(bias[row + i]);";
  }

  std::string apply_activation = GetActivationSnippet(activation_, "output_value_t", "output_element_t");
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform |
                                                      ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation() << GetActivationDeclaration(activation_, "output_value_t", "output_element_t");
  const auto& batch_dims = shader.AddIndices("batch_dims");

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
  const bool has_bias = context.InputCount() > 2;
  std::vector<const Tensor*> inputs(has_bias ? 3 : 2);
  inputs[0] = a;
  inputs[1] = b;
  if (has_bias) {
    const auto* bias = context.Input(2);
    inputs[2] = bias;
  }

  // ComputeMatMul operates on matrices or batched matrices. Promote ONNX MatMul's
  // rank-1 operands to matrix views while keeping the logical output shape above.
  Tensor promoted_a;
  Tensor promoted_b;
  if (a->Shape().NumDimensions() == 1) {
    promoted_a = CreateTensorView(*a, TensorShape({1, a->Shape()[0]}));
    inputs[0] = &promoted_a;
  }
  if (b->Shape().NumDimensions() == 1) {
    promoted_b = CreateTensorView(*b, TensorShape({b->Shape()[0], 1}));
    inputs[1] = &promoted_b;
  }

  return ComputeMatMul(&context, Activation(), inputs, output_tensor,
                       /*is_channels_last=*/true, &compute_cache_, b_is_constant_);
}

Status ComputeMatMul(ComputeContext* context,
                     const Activation& activation, std::vector<const Tensor*>& inputs, Tensor* output_tensor, bool is_channels_last,
                     MatMulComputeCache* cache,
                     bool b_is_constant) {
  const auto* a = inputs[0];
  const auto* b = inputs[1];
  bool has_bias = inputs.size() > 2;
  const TensorShape& logical_a_shape = a->Shape();
  const TensorShape& logical_b_shape = b->Shape();
  ORT_RETURN_IF_NOT(logical_a_shape.NumDimensions() >= 2 && logical_b_shape.NumDimensions() >= 2,
                    "ComputeMatMul expects matrix or batched-matrix inputs.");
  TensorShape a_shape = logical_a_shape;
  TensorShape b_shape = logical_b_shape;

  MatMulComputeHelper helper;
  ORT_THROW_IF_ERROR(helper.Compute(a_shape, b_shape));
  const int64_t batchA =
      a_shape.NumDimensions() > 2 ? a_shape.SizeToDimension(a_shape.NumDimensions() - 2) : 1;
  const int64_t batchB =
      b_shape.NumDimensions() > 2 ? b_shape.SizeToDimension(b_shape.NumDimensions() - 2) : 1;

  TensorShape output_shape = helper.OutputShape();

  // When B is a matrix (batch is 1), we fold batchA into the M dimension for better
  // performance (e.g., [2,3,5] → [1,6,5]).
  if (batchA != 1 && batchB == 1) {
    // dimensions of A: [`batchA` * M, K]
    int64_t batchAndM = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
    TensorShapeVector dims_a = {batchAndM, helper.K()};
    // dimensions of B: [K, N]
    TensorShapeVector dims_b = {helper.K(), helper.N()};

    a_shape = TensorShape(dims_a);
    b_shape = TensorShape(dims_b);
    output_shape = {batchAndM, helper.N()};
  }

  std::unique_ptr<MatMulOptImpl> invocation_impl;
  MatMulOptImpl* subgroup_impl = nullptr;
  if (cache != nullptr) {
    subgroup_impl = cache->GetOrCreateSubgroupMatrixImpl(*context);
  } else {
    invocation_impl = CreateSubgroupMatrixMatMulImpl(*context);
    subgroup_impl = invocation_impl.get();
  }
  if (subgroup_impl != nullptr) {
    bool handled = false;
    ORT_RETURN_IF_ERROR(subgroup_impl->Compute(
        *context, inputs, a_shape, b_shape, output_shape, output_tensor,
        activation, is_channels_last, b_is_constant, cache != nullptr, handled));
    if (handled) {
      return Status::OK();
    }
  }
  const auto naive_info = AnalyzeMatMulNaiveProgram(logical_a_shape, logical_b_shape, is_channels_last);
  if (naive_info.has_value()) {
    MatMulNaiveProgram program{activation, naive_info->output_rank,
                               naive_info->output_number, has_bias, is_channels_last};
    program
        .CacheHint(activation.CacheKey(), std::to_string(naive_info->components),
                   std::to_string(naive_info->a_components),
                   std::to_string(naive_info->output_number), std::to_string(is_channels_last))
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank,
                     naive_info->a_program_shape, narrow<int>(naive_info->a_components)},
                    {b, ProgramTensorMetadataDependency::TypeAndRank,
                     naive_info->b_program_shape, narrow<int>(naive_info->components)}});
    if (has_bias) {
      program.AddInput({inputs[2], ProgramTensorMetadataDependency::Rank,
                        ReduceShapeByComponents(inputs[2]->Shape(), naive_info->bias_components),
                        narrow<int>(naive_info->bias_components)});
    }
    program
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::None,
                      naive_info->output_program_shape, narrow<int>(naive_info->components)}})
        .SetDispatchGroupSize(CeilDiv(naive_info->output_size, 64u))
        .AddIndices(naive_info->outer_dims)
        .AddUniformVariables({{naive_info->output_size}, {naive_info->M}, {naive_info->N}, {naive_info->K}});
    AppendActivationUniformsData(activation, program);
    return context->RunProgram(program);
  }

  if (intel::CanApplyMatMulIntel(*context, helper.M(), helper.N(), helper.K())) {
    return intel::ApplyMatMulIntel(*context, activation, inputs, output_tensor, is_channels_last);
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
  uint32_t dispatch_z = narrow<uint32_t>((static_cast<uint32_t>(batch_size) + MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z * elements_per_thread[2] - 1) /
                                         (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z * elements_per_thread[2]));

  const int components = is_vec4 ? 4 : 1;
  const TensorShape a_shape_temp = CreateMatMulIntermediateShape(outer_dims_a, dim_a_outer, dim_inner, components);
  const TensorShape b_shape_temp = CreateMatMulIntermediateShape(outer_dims_b, dim_inner, dim_b_outer, components);
  const TensorShape output_shape_temp = TensorShape({batch_size, dim_a_outer, dim_b_outer / components});

  ProgramOutput output(output_tensor, ProgramTensorMetadataDependency::Rank, output_shape_temp, components);
  const Tensor* bias = has_bias ? inputs[2] : nullptr;
  bool use_bias_in_matmul = has_bias;
  uint32_t split_dim_inner = 1;
  uint32_t splits_per_batch = 1;

  // Current Split-K implementation relies on atomic operations, which are not deterministic.
  if (!context->KernelContext().GetUseDeterministicCompute()) {
    const SplitKConfig& split_k_config = context->GetSplitKConfig();
    const bool need_split_k = split_k_config.UseSplitK(
        is_vec4, activation.activation_kind_, batch_size, dim_a_outer, dim_b_outer,
        dim_inner, is_channels_last);
    if (need_split_k) {
      ORT_ENFORCE(is_vec4, "Split-K MatMul requires vec4 packing.");

      if (has_bias) {
        ORT_ENFORCE(is_channels_last, "Split-K MatMul only supports channels-last format.");
      }

      // Initialize `output_tensor` with 0 or bias before MatMulProgram with Split-K enabled.
      const auto fill_bias_program = CreateMatMulFillBiasOrZeroBeforeSplitKProgram(bias, output_tensor, /*is_gemm*/ false, /*beta*/ 1.0f, /*bias_components*/ 4, output_shape_temp, narrow<uint32_t>(batch_size));
      ORT_RETURN_IF_ERROR(context->RunProgram(fill_bias_program));

      // `bias` has been handled in the execution of `fill_bias_program` so we don't need to set
      // `bias` again in `MatMulProgram`.
      use_bias_in_matmul = false;

      // With Split-K, `dim_inner` will be split into multiple parts. `dispatch_z` encodes
      // both the split-k index and the batch index: dispatch_z = splits_per_batch * batch_size.
      split_dim_inner = split_k_config.GetSplitDimInner();
      splits_per_batch = (dim_inner + split_dim_inner - 1) / split_dim_inner;
      const uint64_t dispatch_z_u64 = static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(splits_per_batch);
      ORT_ENFORCE(dispatch_z_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                  "dispatch_z exceeds uint32_t range: ", dispatch_z_u64);
      dispatch_z = narrow<uint32_t>(dispatch_z_u64);

      // The output should be declared in atomic types in `MatMulProgram` for the use of atomic
      // built-in functions.
      output.is_atomic = true;
    }
  }

  MatMulProgram matmul_program{activation, use_bias_in_matmul, is_vec4, elements_per_thread, is_channels_last, split_dim_inner};
  matmul_program
      .CacheHint(activation.CacheKey(), absl::StrJoin(elements_per_thread, "-"), std::to_string(is_vec4), components, is_channels_last, split_dim_inner)
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_shape_temp, components},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, b_shape_temp, components}})
      .AddUniformVariables({{dim_a_outer}, {dim_b_outer}, {dim_inner}, {dispatch_x}, {dispatch_y}, {dispatch_z}, {splits_per_batch}})
      .AddIndices(outer_dims)
      .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z)
      .SetWorkgroupSize(MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X, MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y, MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z)
      .AddOutput(std::move(output));
  // Activation uniforms must remain last because definitions and values are matched by index.
  AppendActivationUniformsData(activation, matmul_program);

  if (use_bias_in_matmul) {
    auto bias_components = is_channels_last ? components : 1;
    TensorShape reduced_bias_shape = ReduceShapeByComponents(bias->Shape(), bias_components);
    matmul_program.AddInput({bias, ProgramTensorMetadataDependency::Rank, reduced_bias_shape, bias_components});
  }

  return context->RunProgram(matmul_program);
}

MatMulFillBiasOrZeroBeforeSplitKProgram CreateMatMulFillBiasOrZeroBeforeSplitKProgram(
    const Tensor* bias,
    Tensor* output,
    bool is_gemm,
    float beta,
    uint32_t output_components,
    const TensorShape& output_shape,
    uint32_t batch_size) {
  const bool has_bias = bias != nullptr;
  const bool bias_is_scalar = has_bias ? bias->Shape().Size() == 1 : false;

  MatMulFillBiasOrZeroBeforeSplitKProgram program(is_gemm, has_bias, output_components, bias_is_scalar);

  const uint32_t dim_a_outer = narrow<uint32_t>(output_shape[output_shape.NumDimensions() - 2]);
  const uint32_t dim_b_outer = narrow<uint32_t>(output_shape[output_shape.NumDimensions() - 1]);

  // Fill one value per invocation across all batches.
  const uint64_t total_outputs = static_cast<uint64_t>(batch_size) *
                                 static_cast<uint64_t>(dim_a_outer) *
                                 static_cast<uint64_t>(dim_b_outer);
  const uint64_t dispatch_x_u64 = CeilDiv(total_outputs, static_cast<uint64_t>(WORKGROUP_SIZE));
  ORT_ENFORCE(dispatch_x_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
              "dispatch_x exceeds uint32_t range: ", dispatch_x_u64);
  const uint32_t dispatch_x = narrow<uint32_t>(dispatch_x_u64);

  const uint32_t dim_b_outer_components = narrow<uint32_t>(dim_b_outer * output_components);
  program.CacheHint(is_gemm, has_bias, output_components, bias_is_scalar)
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, output_shape, static_cast<int32_t>(output_components)})
      .AddUniformVariables({{dim_a_outer}, {dim_b_outer_components}, {beta}, {batch_size}})
      .SetDispatchGroupSize(dispatch_x);

  if (has_bias) {
    const TensorShape reduced_bias_shape = ReduceShapeByComponents(bias->Shape(), output_components);
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, reduced_bias_shape, static_cast<int32_t>(output_components)});
  }

  return program;
}

}  // namespace webgpu
}  // namespace onnxruntime
