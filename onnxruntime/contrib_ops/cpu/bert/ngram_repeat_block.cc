// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ngram_repeat_block.h"


namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    NGramRepeatBlock,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("Tid", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NGramRepeatBlock);
}  // namespace contrib
}  // namespace onnxruntime
