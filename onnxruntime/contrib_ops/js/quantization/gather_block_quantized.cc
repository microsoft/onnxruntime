// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_data_types.h"
#include "contrib_ops/js/quantization/gather_block_quantized.h"
namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsepSupportedFloatTypes;

#define ONNX_GATHER_BLOCK_QUANTIZED_KERNELS(T1, Tind)                   \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                    \
      GatherBlockQuantized,                                             \
      kMSDomain, 1,                                                     \
      T1, Tind,                                                         \
      kJsExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())      \
          .TypeConstraint("T2", JsepSupportedFloatTypes())              \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()), \
      GatherBlockQuantized);

ONNX_GATHER_BLOCK_QUANTIZED_KERNELS(UInt4x2, int32_t);
ONNX_GATHER_BLOCK_QUANTIZED_KERNELS(UInt4x2, int64_t);
ONNX_GATHER_BLOCK_QUANTIZED_KERNELS(Int4x2, int32_t);
ONNX_GATHER_BLOCK_QUANTIZED_KERNELS(Int4x2, int64_t);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
