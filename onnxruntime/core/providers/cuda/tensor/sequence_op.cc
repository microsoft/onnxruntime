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

ONNX_OPERATOR_KERNEL_EX(
    SequenceConstruct,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .Alias(0, 0),
    SequenceConstruct);

ONNX_OPERATOR_KERNEL_EX(
    SequenceEmpty,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .Alias(0, 0),
    SequenceEmpty);

ONNX_OPERATOR_KERNEL_EX(
    SequenceLength,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0),
    SequenceLength);

} // namespace cuda
} // namespace onnxruntime


