// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "faster_transformer_bert.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <bert/Bert.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    FasterTransformerBert,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FasterTransformerBert);

Status FasterTransformerBert::ComputeInternal(OpKernelContext* /*context*/) const {
  fastertransformer::BertWeight<float> bert_weights(2, 2, 2);
  return Status::OK();

}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
