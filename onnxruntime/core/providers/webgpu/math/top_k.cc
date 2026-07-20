// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/top_k.h"
#include "core/providers/common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace webgpu {

// Opset 1-9: K is an attribute, no "I" type constraint
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    1, 9,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    TopK);

// Opset 10: K is an input (CPU), largest=true, sorted=true
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    10, 10,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    TopK);

// Opset 11-23: adds largest and sorted attributes
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    11, 23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    TopK);

// Opset 24+
ONNX_OPERATOR_KERNEL_EX(
    TopK,
    kOnnxDomain,
    24,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    TopK);

Status TopKProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("values", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& indices_out = shader.AddOutput("indices", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  const std::string max_float = is_fp16_ ? "65504.0h" : "3.4028234663852886e+38f";
  const std::string pad_value = largest_ ? ("-" + max_float) : max_float;
  // For largest (descending): flip the ascending bit so biggest values come first.
  const std::string asc_expr = largest_ ? "((i / block_size) & 1u) != 0u" : "((i / block_size) & 1u) == 0u";

  // Composite key for stable sort per ONNX spec: "the element with the lower
  // index will appear first" among tied values.
  //
  // For largest=true (descending): key = (value, -index) so lower index sorts first.
  // For largest=false (ascending): key = (value, index) so lower index sorts first.
  // The index tiebreaker flips direction based on largest_ to achieve this.
  const std::string asc_tiebreak = largest_ ? "ia < ib" : "ia > ib";
  const std::string desc_tiebreak = largest_ ? "ia > ib" : "ia < ib";
  const std::string asc_swap = "va > vb || (va == vb && " + asc_tiebreak + ")";
  const std::string desc_swap = "va < vb || (va == vb && " + desc_tiebreak + ")";

  // Declare shared memory for bitonic sort
  shader.AdditionalImplementation()
      << "var<workgroup> shared_vals: array<x_element_t, " << shared_size_ << ">;\n"
      << "var<workgroup> shared_idxs: array<i32, " << shared_size_ << ">;\n";

  shader.MainFunctionBody()
      << "let row = workgroup_idx;\n"
      << "let cols = uniforms.cols;\n"
      << "let k = uniforms.k;\n"
      << "\n"
      // Stride-load all elements into shared memory
      << "for (var idx = local_idx; idx < " << shared_size_ << "u; idx += " << wg_ << "u) {\n"
      << "  if (i32(idx) < cols) {\n"
      << "    shared_vals[idx] = x_element_t(" << input.GetByOffset("u32(i32(row) * cols + i32(idx))") << ");\n"
      << "    shared_idxs[idx] = i32(idx);\n"
      << "  } else {\n"
      << "    shared_vals[idx] = x_element_t(" << pad_value << ");\n"
      << "    shared_idxs[idx] = -1;\n"
      << "  }\n"
      << "}\n"
      << "workgroupBarrier();\n"
      << "\n"
      // Bitonic sort
      << "for (var block_size = 2u; block_size <= " << shared_size_ << "u; block_size <<= 1u) {\n"
      << "  for (var gap = block_size >> 1u; gap > 0u; gap >>= 1u) {\n"
      << "    for (var tid = local_idx; tid < " << shared_size_ / 2 << "u; tid += " << wg_ << "u) {\n"
      << "      let block = tid / gap;\n"
      << "      let pos = tid % gap;\n"
      << "      let i = block * 2u * gap + pos;\n"
      << "      let j = i + gap;\n"
      << "      let asc = " << asc_expr << ";\n"
      << "      let va = shared_vals[i];\n"
      << "      let vb = shared_vals[j];\n"
      << "      let ia = shared_idxs[i];\n"
      << "      let ib = shared_idxs[j];\n"
      << "      let do_swap = select(" << desc_swap << ", " << asc_swap << ", asc);\n"
      << "      if (do_swap) {\n"
      << "        shared_vals[i] = vb;\n"
      << "        shared_vals[j] = va;\n"
      << "        shared_idxs[i] = ib;\n"
      << "        shared_idxs[j] = ia;\n"
      << "      }\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "}\n"
      << "\n"
      // Write top-K results (stride-write for K > wg_size case)
      << "for (var idx = local_idx; idx < u32(k); idx += " << wg_ << "u) {\n"
      << "  let out_offset = u32(i32(row) * k + i32(idx));\n"
      << "  values[out_offset] = shared_vals[idx];\n"
      << "  " << indices_out.SetByOffset("out_offset", "shared_idxs[idx]") << "\n"
      << "}\n";

  return Status::OK();
}

Status TopKInitProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("vals_buf", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("idxs_buf", ShaderUsage::UseUniform);

  const std::string max_float = is_fp16_ ? "65504.0h" : "3.4028234663852886e+38f";
  const std::string pad_value = largest_ ? ("-" + max_float) : max_float;

  shader.MainFunctionBody()
      << "let row = workgroup_idx;\n"
      << "let cols = uniforms.cols;\n"
      << "let pcols = uniforms.padded_cols;\n"
      << "for (var idx = local_idx; idx < u32(pcols); idx += 256u) {\n"
      << "  let out_idx = u32(i32(row) * pcols + i32(idx));\n"
      << "  if (i32(idx) < cols) {\n"
      << "    vals_buf[out_idx] = x_element_t(" << input.GetByOffset("u32(i32(row) * cols + i32(idx))") << ");\n"
      << "    idxs_buf[out_idx] = i32(idx);\n"
      << "  } else {\n"
      << "    vals_buf[out_idx] = x_element_t(" << pad_value << ");\n"
      << "    idxs_buf[out_idx] = -1;\n"
      << "  }\n"
      << "}\n";

  return Status::OK();
}

Status TopKSortStepProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddOutput("vals_buf", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("idxs_buf", ShaderUsage::UseUniform);

  // Same composite-key tiebreaker as shared-memory path
  // Use within-row position (i - base) for direction, not global index
  const std::string asc_expr = largest_ ? "(((i - base) / uniforms.block_size) & 1u) != 0u" : "(((i - base) / uniforms.block_size) & 1u) == 0u";
  const std::string asc_tiebreak = largest_ ? "ia < ib" : "ia > ib";
  const std::string desc_tiebreak = largest_ ? "ia > ib" : "ia < ib";
  const std::string asc_swap = "va > vb || (va == vb && " + asc_tiebreak + ")";
  const std::string desc_swap = "va < vb || (va == vb && " + desc_tiebreak + ")";

  shader.MainFunctionBody()
      << "if (global_idx >= uniforms.total_threads) { return; }\n"
      << "let pcols = uniforms.padded_cols;\n"
      << "let threads_per_row = pcols / 2u;\n"
      << "let row = global_idx / threads_per_row;\n"
      << "let tid = global_idx % threads_per_row;\n"
      << "let gap = uniforms.gap;\n"
      << "let block = tid / gap;\n"
      << "let pos = tid % gap;\n"
      << "let base = row * pcols;\n"
      << "let i = base + block * 2u * gap + pos;\n"
      << "let j = i + gap;\n"
      << "let asc = " << asc_expr << ";\n"
      << "let va = vals_buf[i];\n"
      << "let vb = vals_buf[j];\n"
      << "let ia = idxs_buf[i];\n"
      << "let ib = idxs_buf[j];\n"
      << "let do_swap = select(" << desc_swap << ", " << asc_swap << ", asc);\n"
      << "if (do_swap) {\n"
      << "  vals_buf[i] = vb;\n"
      << "  vals_buf[j] = va;\n"
      << "  idxs_buf[i] = ib;\n"
      << "  idxs_buf[j] = ia;\n"
      << "}\n";

  return Status::OK();
}

Status TopKOutputProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("vals_buf", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("idxs_buf", ShaderUsage::UseUniform);
  shader.AddOutput("values", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& indices_out = shader.AddOutput("indices", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << "let row = workgroup_idx;\n"
      << "let pcols = uniforms.padded_cols;\n"
      << "let k = uniforms.k;\n"
      << "for (var idx = local_idx; idx < u32(k); idx += 256u) {\n"
      << "  let in_idx = u32(i32(row) * pcols + i32(idx));\n"
      << "  let out_idx = u32(i32(row) * k + i32(idx));\n"
      << "  values[out_idx] = vals_buf[in_idx];\n"
      << "  " << indices_out.SetByOffset("out_idx", "idxs_buf[in_idx]") << "\n"
      << "}\n";

  return Status::OK();
}

static uint32_t NextPowerOf2(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

Status TopK::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  const size_t input_rank = input_shape.NumDimensions();

  // Get K value
  int64_t k;
  if (opset_ <= 9) {
    k = attr_k_;
  } else {
    const auto* k_tensor = context.Input(1);
    ORT_ENFORCE(k_tensor != nullptr, "K input tensor is required for TopK opset >= 10");
    ORT_ENFORCE(k_tensor->Shape().Size() == 1, "K tensor should be a scalar or 1-element tensor");
    k = k_tensor->Data<int64_t>()[0];
  }

  ORT_ENFORCE(k >= 0, "k must be non-negative, got ", k);

  // Normalize axis
  const int64_t axis = HandleNegativeAxis(axis_, static_cast<int64_t>(input_rank));

  ORT_ENFORCE(k <= input_shape[onnxruntime::narrow<size_t>(axis)],
              "k (", k, ") must not be greater than the axis dimension (", input_shape[onnxruntime::narrow<size_t>(axis)], ")");

  // Compute output shape
  TensorShape output_shape = input_shape;
  output_shape[onnxruntime::narrow<size_t>(axis)] = k;

  // Handle k=0 case
  if (k == 0) {
    context.Output(0, output_shape);
    context.Output(1, output_shape);
    return Status::OK();
  }

  // Determine if transpose is needed (when axis is not the last dimension)
  const bool is_transpose_required = static_cast<size_t>(axis) < input_rank - 1;

  // Build permutation for transpose
  InlinedVector<size_t> perm(input_rank);
  std::iota(perm.begin(), perm.end(), 0);
  if (is_transpose_required) {
    perm[static_cast<size_t>(axis)] = static_cast<size_t>(input_rank - 1);
    perm[static_cast<size_t>(input_rank - 1)] = static_cast<size_t>(axis);
  }

  // Prepare input (transpose if needed)
  const Tensor* effective_input = input_tensor;
  Tensor transposed_input;
  TensorShape effective_input_shape = input_shape;

  if (is_transpose_required) {
    TensorShapeVector transposed_dims;
    for (auto e : perm) {
      transposed_dims.push_back(input_shape[e]);
    }
    effective_input_shape = TensorShape(transposed_dims);
    transposed_input = context.CreateGPUTensor(input_tensor->DataType(), effective_input_shape);
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, *input_tensor, transposed_input));
    effective_input = &transposed_input;
  }

  // Dimensions for the kernel
  const int64_t cols = effective_input_shape[input_rank - 1];
  const int64_t rows = effective_input_shape.Size() / cols;
  const bool is_fp16 = input_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

  // Prepare output tensors
  Tensor* values_output;
  Tensor* indices_output;
  Tensor transposed_values;
  Tensor transposed_indices;

  TensorShape effective_output_shape = effective_input_shape;
  effective_output_shape[input_rank - 1] = k;

  if (is_transpose_required) {
    transposed_values = context.CreateGPUTensor(input_tensor->DataType(), effective_output_shape);
    transposed_indices = context.CreateGPUTensor(DataTypeImpl::GetType<int64_t>(), effective_output_shape);
    values_output = context.Output(0, output_shape);
    indices_output = context.Output(1, output_shape);
  } else {
    values_output = context.Output(0, output_shape);
    indices_output = context.Output(1, output_shape);
  }

  bool largest = largest_ != 0;
  uint32_t padded = NextPowerOf2(static_cast<uint32_t>(cols));

  // Determine which output tensors the sort kernels write to
  Tensor* sort_values_out = is_transpose_required ? &transposed_values : values_output;
  Tensor* sort_indices_out = is_transpose_required ? &transposed_indices : indices_output;

  if (padded <= 2048) {
    // Small path: single-dispatch shared-memory bitonic sort
    uint32_t wg_size = std::min(padded, 256u);
    TopKProgram program{wg_size, padded, largest, is_fp16};

    program
        .AddInputs({{effective_input, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutputs({{sort_values_out, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutputs({{sort_indices_out, ProgramTensorMetadataDependency::TypeAndRank}})
        .CacheHint(std::to_string(wg_size), std::to_string(padded), largest ? "largest" : "smallest", is_fp16 ? "fp16" : "fp32")
        .SetWorkgroupSize(wg_size)
        .SetDispatchGroupSize(static_cast<uint32_t>(rows))
        .AddUniformVariables({{static_cast<int32_t>(cols)},
                              {static_cast<int32_t>(k)}});

    ORT_RETURN_IF_ERROR(context.RunProgram(program));
  } else {
    // Large path: multi-dispatch global-memory bitonic sort
    uint32_t u_rows = static_cast<uint32_t>(rows);

    // Allocate padded global buffers for sorting
    Tensor vals_buf = context.CreateGPUTensor(input_tensor->DataType(), TensorShape{static_cast<int64_t>(u_rows) * padded});
    Tensor idxs_buf = context.CreateGPUTensor(DataTypeImpl::GetType<int32_t>(), TensorShape{static_cast<int64_t>(u_rows) * padded});

    // 1. Init: load input + padding into global buffers
    TopKInitProgram init_prog{largest, is_fp16};
    init_prog
        .AddInputs({{effective_input, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutputs({{&vals_buf, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutputs({{&idxs_buf, ProgramTensorMetadataDependency::TypeAndRank}})
        .CacheHint(largest ? "largest" : "smallest", is_fp16 ? "fp16" : "fp32")
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(u_rows)
        .AddUniformVariables({{static_cast<int32_t>(cols)},
                              {static_cast<int32_t>(padded)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(init_prog));

    // 2. Bitonic sort steps in global memory
    uint32_t total_threads = u_rows * (padded / 2);
    uint32_t num_wg = (total_threads + 255) / 256;

    for (uint32_t block_size = 2; block_size <= padded; block_size <<= 1) {
      for (uint32_t gap = block_size >> 1; gap > 0; gap >>= 1) {
        TopKSortStepProgram step_prog{largest};
        step_prog
            .AddOutputs({{&vals_buf, ProgramTensorMetadataDependency::TypeAndRank}})
            .AddOutputs({{&idxs_buf, ProgramTensorMetadataDependency::TypeAndRank}})
            .CacheHint(largest ? "largest" : "smallest")
            .SetWorkgroupSize(256)
            .SetDispatchGroupSize(num_wg)
            .AddUniformVariables({{block_size},
                                  {gap},
                                  {padded},
                                  {total_threads}});
        ORT_RETURN_IF_ERROR(context.RunProgram(step_prog));
      }
    }

    // 3. Output: copy top-K from sorted global buffer to output
    TopKOutputProgram out_prog{};
    out_prog
        .AddInputs({{&vals_buf, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddInputs({{&idxs_buf, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutputs({{sort_values_out, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutputs({{sort_indices_out, ProgramTensorMetadataDependency::TypeAndRank}})
        .CacheHint("output")
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(u_rows)
        .AddUniformVariables({{static_cast<int32_t>(padded)},
                              {static_cast<int32_t>(k)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(out_prog));
  }

  // Transpose outputs back if needed
  if (is_transpose_required) {
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, transposed_values, *values_output));
    ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, transposed_indices, *indices_output));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
