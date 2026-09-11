// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/nn/rms_norm.h"
#include "core/providers/webgpu/nn/layer_norm.h"

namespace onnxruntime {
namespace webgpu {

static size_t NormalizeAxis(int64_t axis, size_t tensor_rank) {
  int64_t rank = static_cast<int64_t>(tensor_rank);
  if (axis < -rank && axis >= rank) {
    ORT_THROW("invalid axis: ", axis);
  }
  return onnxruntime::narrow<size_t>(axis < 0 ? axis + rank : axis);
}

static TensorShape GetOverrideShape(const TensorShape& shape, int components) {
  TensorShape override_shape{shape.Size() / components};
  return override_shape;
}

Status RMSNorm::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);

  const auto x_shape = x->Shape();

  const size_t axis = NormalizeAxis(axis_, x_shape.NumDimensions());
  const uint32_t norm_count = onnxruntime::narrow<uint32_t>(x_shape.SizeToDimension(axis));
  const int64_t norm_size = x_shape.SizeFromDimension(axis);
  const int components = GetMaxComponents(norm_size);
  const uint32_t norm_size_vectorized = onnxruntime::narrow<uint32_t>((norm_size + components - 1) / components);

  const auto& scale_shape = scale->Shape();
  const auto scale_size = scale_shape.Size();
  if (scale_shape.NumDimensions() > x_shape.NumDimensions()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Scale and (optional) bias must match X.shape[axis:] or be NumPy-broadcastable to it."
                           " Scale/Bias rank cannot exceed Input rank.");
  }
  if (scale_size != norm_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Size of X.shape()[axis:] == ", norm_size,
                           ". Size of scale must match this. Got scale size of ",
                           scale_size);
  }

  // RMSNormalization outputs: Y (index 0), InvStdDev (index 1, optional)
  auto* y = context.Output(0, x_shape);

  TensorShapeVector inv_std_dev_dim;
  for (size_t i = 0; i < x_shape.NumDimensions(); ++i) {
    if (i < axis) {
      inv_std_dev_dim.push_back(x_shape[i]);
    } else {
      inv_std_dev_dim.push_back(1);
    }
  }
  TensorShape inv_std_dev_shape(inv_std_dev_dim);
  auto* inv_std_dev = context.Output(1, inv_std_dev_shape);

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  // Check if we should use split norm dimension optimization
  const bool split_norm_dim = norm_size % 512 == 0 && norm_count == 1;

  // Reuse LayerNormProgram with simplified=true, has_bias=false, no mean output
  LayerNormProgram program{/*has_bias=*/false, /*simplified=*/true, /*has_mean_output=*/false,
                           /*has_inv_std_dev_output=*/inv_std_dev != nullptr, split_norm_dim};

  program.CacheHint(components, /*simplified=*/true, split_norm_dim)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, GetOverrideShape(x->Shape(), components), components}})
      .AddInputs(
          {{scale, ProgramTensorMetadataDependency::Type, GetOverrideShape(scale->Shape(), components), components}})
      .AddOutputs({{y, ProgramTensorMetadataDependency::None, GetOverrideShape(y->Shape(), components), components}})
      .AddUniformVariables({
          {static_cast<uint32_t>(components)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(norm_count)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(norm_size)},
      })
      .AddUniformVariables({
          {static_cast<uint32_t>(norm_size_vectorized)},
      })
      .AddUniformVariables({
          {static_cast<float>(epsilon_)},
      });

  if (split_norm_dim) {
    const uint32_t workgroup_size_x = 128;
    const uint32_t dispatch_size_x = onnxruntime::narrow<uint32_t>(norm_size / (workgroup_size_x * components));
    program.SetDispatchGroupSize(dispatch_size_x, 1, 1)
        .SetWorkgroupSize(workgroup_size_x);
  } else {
    program.SetDispatchGroupSize(norm_count);
  }

  if (inv_std_dev != nullptr) {
    program.AddOutputs({{inv_std_dev, ProgramTensorMetadataDependency::None}});
  }

  return context.RunProgram(program);
}

ONNX_OPERATOR_KERNEL_EX(RMSNormalization, kOnnxDomain, 23, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", WebGpuSupportedFloatTypes())
                            .TypeConstraint("V", WebGpuSupportedFloatTypes()),
                        RMSNorm);

}  // namespace webgpu
}  // namespace onnxruntime
