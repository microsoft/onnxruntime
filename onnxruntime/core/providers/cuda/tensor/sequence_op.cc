// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_op.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SequenceAt,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()})
        .Alias(0, 0),
    SequenceAt);

} // namespace cuda
} // namespace onnxruntime


