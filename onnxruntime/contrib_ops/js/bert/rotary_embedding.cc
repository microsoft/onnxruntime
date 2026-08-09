// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "rotary_embedding.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsepSupportedFloatTypes;

ONNX_OPERATOR_KERNEL_EX(RotaryEmbedding, kMSDomain, 1, kJsExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", JsepSupportedFloatTypes())
                            .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()),
                        RotaryEmbedding);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
