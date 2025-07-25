
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/string_macros.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/pool.h"

#include <vector>

namespace onnxruntime {
namespace webgpu {

namespace {

std::vector<uint32_t> NarrowToU32(const TensorShapeVector& shape) {
  std::vector<uint32_t> result;
  result.reserve(shape.size());
  for (auto dim : shape) {
    result.push_back(static_cast<uint32_t>(dim));
  }
  return result;
}

}  // namespace

#define POOLING_KERNEL(op_name, domain, is_nhwc, pool_type, since_version)                                \
  ONNX_OPERATOR_KERNEL_EX(op_name, domain, since_version, kWebGpuExecutionProvider,                       \
                          (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
                          Pool<pool_type, is_nhwc>);

#define POOLING_KERNEL_VERSIONED(op_name, domain, is_nhwc, pool_type, since_version, end_version)                   \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(op_name, domain, since_version, end_version, kWebGpuExecutionProvider,          \
                                    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
                                    Pool<pool_type, is_nhwc>);

#define POOLING_KERNEL_WITH_INDICES(op_name, domain, is_nhwc, pool_type, since_version)     \
  ONNX_OPERATOR_KERNEL_EX(op_name, domain, since_version, kWebGpuExecutionProvider,         \
                          (*KernelDefBuilder::Create())                                     \
                              .TypeConstraint("T", WebGpuSupportedFloatTypes())             \
                              .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
                          Pool<pool_type, is_nhwc>);

#define POOLING_KERNEL_VERSIONED_WITH_INDICES(op_name, domain, is_nhwc, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(op_name, domain, since_version, end_version, kWebGpuExecutionProvider,     \
                                    (*KernelDefBuilder::Create())                                              \
                                        .TypeConstraint("T", WebGpuSupportedFloatTypes())                      \
                                        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),          \
                                    Pool<pool_type, is_nhwc>);

POOLING_KERNEL_VERSIONED(AveragePool, kOnnxDomain, false, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, kMSInternalNHWCDomain, true, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, kOnnxDomain, false, AveragePool, 10, 10)
POOLING_KERNEL_VERSIONED(AveragePool, kMSInternalNHWCDomain, true, AveragePool, 10, 10)
POOLING_KERNEL(AveragePool, kOnnxDomain, false, AveragePool, 11)
POOLING_KERNEL(AveragePool, kMSInternalNHWCDomain, true, AveragePool, 11)
POOLING_KERNEL(GlobalAveragePool, kOnnxDomain, false, AveragePool, 1)
POOLING_KERNEL(GlobalAveragePool, kMSInternalNHWCDomain, true, AveragePool, 1)

POOLING_KERNEL_VERSIONED(MaxPool, kOnnxDomain, false, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, kMSInternalNHWCDomain, true, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 11, 11)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 11, 11)
POOLING_KERNEL_WITH_INDICES(MaxPool, kOnnxDomain, false, MaxPool<8>, 12)
POOLING_KERNEL_WITH_INDICES(MaxPool, kMSInternalNHWCDomain, true, MaxPool<8>, 12)
POOLING_KERNEL(GlobalMaxPool, kOnnxDomain, false, MaxPool<1>, 1)
POOLING_KERNEL(GlobalMaxPool, kMSInternalNHWCDomain, true, MaxPool<1>, 1)

Status PoolProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  // Declare and initialize the variables needed.
  std::string var_decl_code;
  // Process each element in the pooling window.
  std::string sampling_code;
  // Calculate the output value for each pooling window.
  std::string downsampling_code;

  constexpr const size_t kStringInitialSize = 128;
  if (is_max_pool_) {
    std::string f16_min = "f16(-65504)";

    SS(f32_min_ss, kStringInitialSize);
    f32_min_ss << "f32(" << std::numeric_limits<float>::lowest() << ")";
    std::string f32_min = SS_GET(f32_min_ss);

    SS(var_decl_ss, kStringInitialSize);
    var_decl_ss << "  var value = " << (is_float16_ ? f16_min : f32_min) << ";\n";
    var_decl_code = SS_GET(var_decl_ss);

    sampling_code = "      value = max(value, x_val);\n";
    if (are_small_output_big_kernel_) {
      downsampling_code = "  sum_or_max_shared[local_idx] = value;\n";
    }
  } else {
    SS(var_decl_ss, kStringInitialSize);
    var_decl_ss << "  var value = " << (is_float16_ ? "f16(0)" : "f32(0)") << ";\n";
    if (!count_include_pad_) {
      var_decl_ss << "  var count = u32(0);\n";
    } else {
      var_decl_ss << "  var count = uniforms.kernel_size;\n";
    }
    var_decl_code = SS_GET(var_decl_ss);

    SS(sampling_ss, kStringInitialSize);
    sampling_ss << "      value += x_val;\n";
    if (!count_include_pad_) {
      sampling_ss << "      count++;\n";
    }
    sampling_code = SS_GET(sampling_ss);

    SS(downsampling_ss, kStringInitialSize);
    if (are_small_output_big_kernel_) {
      downsampling_ss << "  sum_or_max_shared[local_idx] = value;\n"
                      << "  count_shared[local_idx] = count;\n";
    } else {
      downsampling_ss << "  value /= " << (is_float16_ ? "f16" : "f32") << "(count);\n";
    }
    downsampling_code = SS_GET(downsampling_ss);
  }

  const auto kernel_rank = kernel_shape_.size();
  const auto pads_rank = kernel_shape_.size() * 2;
  // The dimension index for H or D1
  const auto data_dim_begin = is_nhwc_ ? 1 : 2;
  // The dimension index after W or Dn
  auto data_dim_end = input.Rank();
  data_dim_end = is_nhwc_ ? data_dim_end - 1 : data_dim_end;

  std::string sum_or_max_shared;
  if (are_small_output_big_kernel_) {
    shader.AdditionalImplementation()
        << "var<workgroup> sum_or_max_shared : array<" << (is_float16_ ? "f16" : "f32") << ",workgroup_size_x >;\n"
        << (!is_max_pool_ ? "var<workgroup> count_shared : array<u32, workgroup_size_x>;\n" : "");

    SS(shared_ss, 512);
    std::string sum_or_max_shared_op;
    std::string count_shared_op;
    if (is_max_pool_) {
      sum_or_max_shared_op = "sum_or_max_shared[local_idx] = max(sum_or_max_shared[local_idx], sum_or_max_shared[local_idx + reduce_size]);\n";
    } else {
      sum_or_max_shared_op = "sum_or_max_shared[local_idx] += sum_or_max_shared[local_idx + reduce_size];\n";
      count_shared_op = "count_shared[local_idx] += count_shared[local_idx + reduce_size];\n";
    }

    shared_ss << "  workgroupBarrier();\n"
              << "  var reduce_size : u32 = workgroup_size_x;\n"
              << "  for (var curr_size = reduce_size >> 1;  curr_size > 0; curr_size = reduce_size >> 1) {\n"
              << "    reduce_size = curr_size + (reduce_size & 1);\n"
              << "    if (local_idx < curr_size) {\n"
              << "      " << sum_or_max_shared_op
              << "      " << count_shared_op
              << "    }\n"
              << "    workgroupBarrier();\n"
              << "  }\n";
    sum_or_max_shared = SS_GET(shared_ss);
  }
  std::string kernel_loop_decl_code = are_small_output_big_kernel_ ? "  for (var i: u32 = local_idx; i < uniforms.kernel_size; i += workgroup_size_x) {\n" : "  for (var i: u32 = 0; i < uniforms.kernel_size; i++) {\n";

  SS(output_ss, kStringInitialSize);
  if (are_small_output_big_kernel_) {
    output_ss << "  if (local_idx == 0) {\n"
              << "    value = sum_or_max_shared[0]" << (!is_max_pool_ ? (is_float16_ ? " / f16(count_shared[0])" : " / f32(count_shared[0])") : "") << ";\n"
              << "    " << output.SetByOffset("workgroup_idx", "value") << ";\n"
              << "  }\n";
  } else {
    output_ss << "  " << output.SetByOffset("global_idx", "value") << ";\n";
  }
  std::string output_code = SS_GET(output_ss);

  auto& body = shader.MainFunctionBody();
  body << (are_small_output_big_kernel_ ? "" : shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size"))
       << "  let y_indices = " << output.OffsetToIndices((are_small_output_big_kernel_ ? "workgroup_idx" : "global_idx")) << ";\n"
       << "  var x_indices = y_indices;\n"
       << "  var k_indices: array<u32, " << kernel_rank << ">;\n"
       << var_decl_code
       << kernel_loop_decl_code
       << "    var offset = i;\n"
       // ---- Compute offset to indices in pooling window.
       << "    for (var j = 0; j < " << kernel_rank << "; j++) {\n"
       << "      k_indices[j] = offset / " << GetElementAt("uniforms.kernel_strides", "j", kernel_rank) << ";\n"
       << "      offset = offset % " << GetElementAt("uniforms.kernel_strides", "j", kernel_rank) << ";\n"
       << "    }\n"
       // ---- Apply dilations in pooling window.
       << "    for (var j = 0; j < " << kernel_rank << "; j++) {\n"
       << "      k_indices[j] *= " << GetElementAt("uniforms.dilations", "j", kernel_rank) << ";\n"
       << "    }\n"
       << "    var is_pad = false;\n"
       // ---- Compute x_indices in each data dimension
       << "    for (var j = " << data_dim_begin << "; j < " << data_dim_end << "; j++) {\n"
       << "      let d_idx = j - " << data_dim_begin << ";\n"
       << "      x_indices[j] = y_indices[j] * " << GetElementAt("uniforms.strides", "d_idx", kernel_rank) << ";\n"
       << "      x_indices[j] += k_indices[d_idx];\n"
       << "      x_indices[j] -= " << GetElementAt("uniforms.pads", "d_idx", pads_rank) << ";\n"
       << "      let j_dim_len = " << input.IndicesGet("uniforms.input_shape", "j") << ";\n"
       // ------ Check if x_indices[j] is out of bounds to handle padding.
       << "      if (x_indices[j] < 0 || x_indices[j] >= j_dim_len) {\n"
       << "        is_pad = true;\n"
       << "        break;\n"
       << "      }\n"
       << "    }\n"
       << "    if (!is_pad) {\n"
       << "      let x_val = " << input.GetByIndices("x_indices") << ";\n"
       << sampling_code
       << "    }\n"
       << "  }\n"
       << downsampling_code
       << sum_or_max_shared
       << output_code;

  return Status::OK();
}

template <typename PoolType, bool is_nhwc>
Status Pool<PoolType, is_nhwc>::ComputeInternal(ComputeContext& context) const {
  // TODO: support 'column major' storage_order.
  ORT_RETURN_IF_NOT(pool_attrs_.storage_order == 0, "Using column major storage_order is not supported yet.");

  // TODO: support 'Indices' output.
  ORT_RETURN_IF_NOT(context.OutputCount() == 1, "The Indices output is not supported yet.");

  const auto* X = context.Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto input_shape = x_shape.AsShapeVector();
  ORT_RETURN_IF_NOT(input_shape.size() >= 3, "Input dimension cannot be less than 3.");

  auto kernel_shape = pool_attrs_.kernel_shape;
  auto strides = pool_attrs_.strides;
  auto pads = pool_attrs_.pads;
  auto dilations = pool_attrs_.dilations;
  // Global pooling is equivalent to having the kernel size equal to the spatial dimension of input tensor.
  if (pool_attrs_.global_pooling) {
    if (!is_nhwc) {
      kernel_shape.assign(input_shape.begin() + 2, input_shape.end());
    } else {
      kernel_shape.assign(input_shape.begin() + 1, input_shape.end() - 1);
    }
    // No padding.
    pads.assign(2 * kernel_shape.size(), 0);
    // Stride of 1.
    strides.assign(kernel_shape.size(), 1);
    // Dilation of 1.
    dilations.assign(kernel_shape.size(), 1);
  }

  // Calculate the output shape
  const auto out_channel = x_shape[is_nhwc ? input_shape.size() - 1 : 1];
  const auto output_shape = pool_attrs_.SetOutputSize(x_shape, out_channel, &pads, is_nhwc);
  Tensor* Y = context.Output(0, output_shape);

  std::vector<uint32_t> kernel_strides(kernel_shape.size());
  ORT_ENFORCE(kernel_shape.size() > 0, "kernel_shape must have at least one element.");
  // Calculate the kernel element strides for each dimension in reverse order. For example:
  //   kernel_shape = [3, 2], kernel_strides = [2, 1]
  //   kernel_shape = [2, 3, 2], kernel_strides = [6, 2, 1]
  for (size_t i = kernel_shape.size(); i > 0; --i) {
    if (i == kernel_shape.size()) {
      kernel_strides[i - 1] = 1;
    } else {
      kernel_strides[i - 1] = kernel_strides[i] * gsl::narrow_cast<uint32_t>(kernel_shape[i]);
    }
  }

  bool is_max_pool = false;
  if constexpr (PoolType::type == onnxruntime::PoolType::kMaxPool) {
    is_max_pool = true;
  } else if constexpr (PoolType::type != onnxruntime::PoolType::kAveragePool) {
    ORT_NOT_IMPLEMENTED("Unsupported PoolType.");
  }
  bool is_float16 = X->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  bool count_include_pad = pool_attrs_.count_include_pad;

  // Number of elements
  uint32_t output_size = gsl::narrow_cast<uint32_t>(Y->Shape().Size());
  uint32_t kernel_size = gsl::narrow_cast<uint32_t>(TensorShape{kernel_shape}.Size());

  const auto pads_u32 = NarrowToU32(pads);
  const auto strides_u32 = NarrowToU32(strides);
  const auto dilations_u32 = NarrowToU32(dilations);

  bool are_small_output_big_kernel = output_size <= 128 && kernel_size >= 128;
  PoolProgram program{is_max_pool, is_nhwc, kernel_shape, is_float16, count_include_pad, are_small_output_big_kernel};

  program.CacheHint(kernel_shape.size(), is_max_pool, is_nhwc, is_float16, count_include_pad, are_small_output_big_kernel)
      .AddInputs({{X, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{Y}})
      .AddUniformVariables({output_size, kernel_size,
                            gsl::span<const uint32_t>(kernel_strides.data(), kernel_strides.size()),
                            gsl::span<const uint32_t>(pads_u32.data(), pads_u32.size()),
                            gsl::span<const uint32_t>(strides_u32.data(), strides_u32.size()),
                            gsl::span<const uint32_t>(dilations_u32.data(), dilations_u32.size())});

  if (are_small_output_big_kernel) {
    program.SetWorkgroupSize(128)
        .SetDispatchGroupSize(output_size);
  } else {
    program.SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  }

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
