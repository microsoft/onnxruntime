// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_data_types.h"
#include "core/providers/js/operators/layer_norm.h"

namespace onnxruntime {
namespace contrib {
namespace js {

// LayerNormalization used to be a contrib op
// that (incorrectly) used kOnnxDomain so we need to version it
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    LayerNormalization,
    kOnnxDomain,
    1,
    16,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", onnxruntime::js::JsepSupportedFloatTypes())
        .TypeConstraint("U", onnxruntime::js::JsepSupportedFloatTypes()),
    onnxruntime::js::LayerNorm<false>);

ONNX_OPERATOR_KERNEL_EX(
    SimplifiedLayerNormalization,
    kOnnxDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", onnxruntime::js::JsepSupportedFloatTypes())
        .TypeConstraint("U", onnxruntime::js::JsepSupportedFloatTypes()),
    onnxruntime::js::LayerNorm<true>);

#define REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL(T)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      SimplifiedLayerNormalization,                                  \
      kMSDomain,                                                     \
      1,                                                             \
      T,                                                             \
      kJsExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()),    \
      onnxruntime::js::LayerNorm<true>);

REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL(float)
REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL(MLFloat16)

#undef REGISTER_SIMPLIFIED_LAYER_NORM_KERNEL

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
