
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LayerNormalization,
    kOnnxDomain,
    1,
    16,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    onnxruntime::webgpu::LayerNorm<false>);

ONNX_OPERATOR_KERNEL_EX(
    SimplifiedLayerNormalization,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    onnxruntime::webgpu::LayerNorm<true>);

#define REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL(T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      SimplifiedLayerNormalization,                                  \
      kMSDomain,                                                     \
      1,                                                             \
      T,                                                             \
      kWebGpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),    \
      onnxruntime::webgpu::LayerNorm<true>);

REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL(float)
REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL(MLFloat16)

#undef REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
