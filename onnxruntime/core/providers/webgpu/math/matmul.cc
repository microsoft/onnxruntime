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
#include "core/providers/webgpu/vendor/intel/math/matmul_algorithm_scheduler.h"
#include "core/providers/webgpu/vendor/intel/math/matmul.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

std::unique_ptr<MatMulOptImpl> CreateSubgroupMatrixMatMulImpl(const ComputeContextBase& context);

MatMulOptImpl* MatMulOptImplCache::GetOrCreate(const ComputeContextBase& context) {
  std::call_once(subgroup_impl_init_flag_, [&]() {
    subgroup_impl_ = CreateSubgroupMatrixMatMulImpl(context);
  });
  return subgroup_impl_.get();
}

const MatMulAlgorithmScheduler& MatMulOptImplCache::GetOrCreateScheduler(const ComputeContextBase& context) {
  std::call_once(scheduler_init_flag_, [&]() {
    if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
      scheduler_ = std::make_unique<intel::IntelMatMulAlgorithmScheduler>();
    } else {
      scheduler_ = std::make_unique<MatMulAlgorithmScheduler>();
    }
  });
  return *scheduler_;
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
                       /*is_channels_last=*/true, compute_cache_, b_is_constant_);
}

static Status ApplyMatMulNaive(ComputeContext& context,
                               const Activation& activation,
                               const std::vector<const Tensor*>& inputs,
                               Tensor* output_tensor,
                               bool is_channels_last,
                               const MatMulComputeHelper& helper) {
  const auto* a = inputs[0];
  const auto* b = inputs[1];
  const bool has_bias = inputs.size() > 2;
  const uint32_t m = narrow<uint32_t>(helper.M());
  const uint32_t n = narrow<uint32_t>(helper.N());
  const uint32_t k = narrow<uint32_t>(helper.K());
  const int components = GetMaxComponents(n);
  const int a_components = GetMaxComponents(k);
  const int64_t output_number = GetMaxComponents(m);
  const TensorShape& logical_output_shape = helper.OutputShape();
  const size_t output_rank = logical_output_shape.NumDimensions();
  const TensorShape outer_dims =
      output_rank > 2 ? logical_output_shape.Slice(0, output_rank - 2) : TensorShape({});
  const int64_t output_rows = a->Shape()[a->Shape().NumDimensions() - 2];
  const TensorShape output_program_shape{
      outer_dims.Size(), output_rows, n / components};
  const uint32_t output_size =
      narrow<uint32_t>(logical_output_shape.Size() / components / output_number);

  MatMulNaiveProgram program{activation, output_rank, output_number, has_bias, is_channels_last};
  program
      .CacheHint(activation.CacheKey(), std::to_string(components),
                 std::to_string(a_components), std::to_string(output_number),
                 std::to_string(is_channels_last))
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_components},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (has_bias) {
    const int bias_components = is_channels_last ? components : 1;
    program.AddInput({inputs[2], ProgramTensorMetadataDependency::Rank, bias_components});
  }
  program
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::None,
                    output_program_shape, components}})
      .SetDispatchGroupSize(CeilDiv(output_size, 64u))
      .AddIndices(outer_dims)
      .AddUniformVariables({{output_size}, {m}, {n}, {k}});
  AppendActivationUniformsData(activation, program);
  return context.RunProgram(program);
}

static bool ShouldUsePackedSplitK(ComputeContext& context,
                                  const Activation& activation,
                                  const std::vector<const Tensor*>& inputs,
                                  bool is_channels_last,
                                  const MatMulComputeHelper& helper) {
  if (context.KernelContext().GetUseDeterministicCompute()) {
    return false;
  }

  TensorShape a_shape = inputs[0]->Shape();
  TensorShape b_shape = inputs[1]->Shape();
  TensorShape output_shape = helper.OutputShape();
  const int64_t batch_a =
      a_shape.NumDimensions() > 2 ? a_shape.SizeToDimension(a_shape.NumDimensions() - 2) : 1;
  const int64_t batch_b =
      b_shape.NumDimensions() > 2 ? b_shape.SizeToDimension(b_shape.NumDimensions() - 2) : 1;
  if (batch_a != 1 && batch_b == 1) {
    const int64_t batch_and_m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
    a_shape = TensorShape({batch_and_m, helper.K()});
    b_shape = TensorShape({helper.K(), helper.N()});
    output_shape = TensorShape({batch_and_m, helper.N()});
  }

  const int64_t batch_size = output_shape.NumDimensions() > 2
                                 ? output_shape.SizeToDimension(output_shape.NumDimensions() - 2)
                                 : 1;
  const uint32_t m = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);
  const uint32_t k = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
  const uint32_t n = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 1]);
  const bool is_vec4 = k % 4 == 0 && n % 4 == 0;
  return context.GetSplitKConfig().UseSplitK(
      is_vec4, activation.activation_kind_, batch_size, m, n, k, is_channels_last);
}

static Status ApplyMatMulPacked(ComputeContext& context,
                                const Activation& activation,
                                const std::vector<const Tensor*>& inputs,
                                Tensor* output_tensor,
                                bool is_channels_last,
                                const MatMulComputeHelper& helper,
                                bool use_split_k) {
  const auto* a = inputs[0];
  const auto* b = inputs[1];
  const bool has_bias = inputs.size() > 2;
  TensorShape a_shape = a->Shape();
  TensorShape b_shape = b->Shape();
  TensorShape output_shape = helper.OutputShape();
  const int64_t batch_a =
      a_shape.NumDimensions() > 2 ? a_shape.SizeToDimension(a_shape.NumDimensions() - 2) : 1;
  const int64_t batch_b =
      b_shape.NumDimensions() > 2 ? b_shape.SizeToDimension(b_shape.NumDimensions() - 2) : 1;

  if (batch_a != 1 && batch_b == 1) {
    const int64_t batch_and_m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
    a_shape = TensorShape({batch_and_m, helper.K()});
    b_shape = TensorShape({helper.K(), helper.N()});
    output_shape = TensorShape({batch_and_m, helper.N()});
  }

  const TensorShape outer_dims_a = a_shape.NumDimensions() > 2
                                       ? a_shape.Slice(0, a_shape.NumDimensions() - 2)
                                       : TensorShape({});
  const TensorShape outer_dims_b = b_shape.NumDimensions() > 2
                                       ? b_shape.Slice(0, b_shape.NumDimensions() - 2)
                                       : TensorShape({});
  const TensorShape outer_dims = output_shape.NumDimensions() > 2
                                     ? output_shape.Slice(0, output_shape.NumDimensions() - 2)
                                     : TensorShape({});
  const int64_t batch_size = outer_dims.Size();
  const uint32_t dim_a_outer = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 2]);
  const uint32_t dim_inner = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
  const uint32_t dim_b_outer = narrow<uint32_t>(b_shape[b_shape.NumDimensions() - 1]);
  const bool is_vec4 = dim_inner % 4 == 0 && dim_b_outer % 4 == 0;

  InlinedVector<int64_t> elements_per_thread = dim_a_outer <= 8
                                                   ? InlinedVector<int64_t>({4, 1, 1})
                                                   : InlinedVector<int64_t>({4, 4, 1});
  const uint32_t dispatch_x = narrow<uint32_t>(
      (dim_b_outer + MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X * elements_per_thread[0] - 1) /
      (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X * elements_per_thread[0]));
  const uint32_t dispatch_y = narrow<uint32_t>(
      (dim_a_outer + MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y * elements_per_thread[1] - 1) /
      (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y * elements_per_thread[1]));
  uint32_t dispatch_z = narrow<uint32_t>(
      (static_cast<uint32_t>(batch_size) +
       MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z * elements_per_thread[2] - 1) /
      (MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z * elements_per_thread[2]));

  const int components = is_vec4 ? 4 : 1;
  const TensorShape a_shape_temp =
      CreateMatMulIntermediateShape(outer_dims_a, dim_a_outer, dim_inner, components);
  const TensorShape b_shape_temp =
      CreateMatMulIntermediateShape(outer_dims_b, dim_inner, dim_b_outer, components);
  const TensorShape output_shape_temp{batch_size, dim_a_outer, dim_b_outer / components};
  ProgramOutput output(output_tensor, ProgramTensorMetadataDependency::Rank, output_shape_temp, components);
  const Tensor* bias = has_bias ? inputs[2] : nullptr;
  bool use_bias_in_matmul = has_bias;
  uint32_t split_dim_inner = 1;
  uint32_t splits_per_batch = 1;

  if (use_split_k) {
    ORT_RETURN_IF(context.KernelContext().GetUseDeterministicCompute(),
                  "MatMul algorithm packed_split_k does not support deterministic compute.");
    ORT_RETURN_IF_NOT(context.GetSplitKConfig().GetSplitDimInner() != 0,
                      "MatMul algorithm packed_split_k is not configured for this adapter.");
    ORT_RETURN_IF_NOT(is_vec4,
                      "MatMul algorithm packed_split_k requires vec4 packing.");
    ORT_RETURN_IF_NOT(activation.activation_kind_ == ActivationKind::None,
                      "MatMul algorithm packed_split_k does not support a fused activation.");
    ORT_RETURN_IF_NOT(!has_bias || is_channels_last,
                      "MatMul algorithm packed_split_k requires channels-last bias layout.");

    const auto fill_bias_program = CreateMatMulFillBiasOrZeroBeforeSplitKProgram(
        bias, output_tensor, /*is_gemm=*/false, /*beta=*/1.0f,
        /*output_components=*/4, output_shape_temp, narrow<uint32_t>(batch_size));
    ORT_RETURN_IF_ERROR(context.RunProgram(fill_bias_program));
    use_bias_in_matmul = false;
    split_dim_inner = context.GetSplitKConfig().GetSplitDimInner();
    splits_per_batch = (dim_inner + split_dim_inner - 1) / split_dim_inner;
    const uint64_t dispatch_z_u64 =
        static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(splits_per_batch);
    ORT_RETURN_IF_NOT(dispatch_z_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                      "MatMul algorithm packed_split_k dispatch_z exceeds uint32_t range: ", dispatch_z_u64);
    dispatch_z = narrow<uint32_t>(dispatch_z_u64);
    output.is_atomic = true;
  }

  MatMulProgram program{activation, use_bias_in_matmul, is_vec4, elements_per_thread,
                        is_channels_last, split_dim_inner};
  program
      .CacheHint(activation.CacheKey(), absl::StrJoin(elements_per_thread, "-"),
                 std::to_string(is_vec4), components, is_channels_last, split_dim_inner)
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_shape_temp, components},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, b_shape_temp, components}})
      .AddUniformVariables({{dim_a_outer}, {dim_b_outer}, {dim_inner}, {dispatch_x}, {dispatch_y}, {dispatch_z}, {splits_per_batch}})
      .AddIndices(outer_dims)
      .SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z)
      .SetWorkgroupSize(MatMul::MATMUL_PACKED_WORKGROUP_SIZE_X,
                        MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Y,
                        MatMul::MATMUL_PACKED_WORKGROUP_SIZE_Z)
      .AddOutput(std::move(output));
  AppendActivationUniformsData(activation, program);

  if (use_bias_in_matmul) {
    const int bias_components = is_channels_last ? components : 1;
    const TensorShape reduced_bias_shape = ReduceShapeByComponents(bias->Shape(), bias_components);
    program.AddInput({bias, ProgramTensorMetadataDependency::Rank, reduced_bias_shape, bias_components});
  }

  return context.RunProgram(program);
}

Status ComputeMatMul(ComputeContext* context,
                     const Activation& activation, std::vector<const Tensor*>& inputs, Tensor* output_tensor,
                     bool is_channels_last, MatMulOptImplCache& cache,
                     bool b_is_constant) {
  const auto* a = inputs[0];
  const auto* b = inputs[1];
  const bool has_bias = inputs.size() > 2;
  const TensorShape& logical_a_shape = a->Shape();
  const TensorShape& logical_b_shape = b->Shape();
  ORT_RETURN_IF_NOT(logical_a_shape.NumDimensions() >= 2 && logical_b_shape.NumDimensions() >= 2,
                    "ComputeMatMul expects matrix or batched-matrix inputs.");

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(logical_a_shape, logical_b_shape));

  MatMulOptImpl* subgroup_impl = cache.GetOrCreate(*context);
  const bool can_use_subgroup_matrix =
      subgroup_impl != nullptr &&
      subgroup_impl->CanApply(*context, inputs, is_channels_last, b_is_constant);
  const bool has_intel_subgroup_capability = intel::HasMatMulIntelCapability(*context);
  const bool use_split_k =
      ShouldUsePackedSplitK(*context, activation, inputs, is_channels_last, helper);

  MatMulAlgorithmSelectionParams selection_params{};
  selection_params.m = helper.M();
  selection_params.n = helper.N();
  selection_params.k = helper.K();
  selection_params.can_use_subgroup_matrix = can_use_subgroup_matrix;
  selection_params.has_intel_subgroup_capability = has_intel_subgroup_capability;
  selection_params.use_split_k = use_split_k;

  const MatMulAlgorithm algorithm =
      cache.GetOrCreateScheduler(*context).Select(selection_params, context->ForcedMatMulAlgorithm());

  MatMulAlgorithmPrerequisites prerequisites{};
  prerequisites.can_use_subgroup_matrix = can_use_subgroup_matrix;
  prerequisites.has_intel_subgroup_capability = has_intel_subgroup_capability;
  prerequisites.split_k_configured = context->GetSplitKConfig().GetSplitDimInner() != 0;
  prerequisites.deterministic_compute = context->KernelContext().GetUseDeterministicCompute();
  prerequisites.is_vec4 = helper.K() % 4 == 0 && helper.N() % 4 == 0;
  prerequisites.has_fused_activation = activation.activation_kind_ != ActivationKind::None;
  prerequisites.split_k_bias_layout_supported = !has_bias || is_channels_last;
  ORT_RETURN_IF_NOT(MeetsMatMulAlgorithmPrerequisites(algorithm, prerequisites),
                    "MatMul algorithm ", MatMulAlgorithmName(algorithm),
                    " does not support these inputs or this device.");

  switch (algorithm) {
    case MatMulAlgorithm::SubgroupMatrix:
      ORT_RETURN_IF_NOT(subgroup_impl != nullptr,
                        "MatMul algorithm subgroup_matrix is unavailable.");
      return subgroup_impl->Compute(
          *context, inputs, output_tensor, activation, is_channels_last, b_is_constant);
    case MatMulAlgorithm::Naive:
      return ApplyMatMulNaive(
          *context, activation, inputs, output_tensor, is_channels_last, helper);
    case MatMulAlgorithm::IntelSubgroup:
      return intel::ApplyMatMulIntel(
          *context, activation, inputs, output_tensor, is_channels_last);
    case MatMulAlgorithm::Packed:
      return ApplyMatMulPacked(
          *context, activation, inputs, output_tensor, is_channels_last, helper,
          /*use_split_k=*/false);
    case MatMulAlgorithm::PackedSplitK:
      return ApplyMatMulPacked(
          *context, activation, inputs, output_tensor, is_channels_last, helper,
          /*use_split_k=*/true);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown MatMul algorithm.");
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
