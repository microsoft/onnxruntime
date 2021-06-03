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
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
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
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .Alias(0, 0),
    SequenceConstruct);

ONNX_OPERATOR_KERNEL_EX(
    SequenceEmpty,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .Alias(0, 0),
    SequenceEmpty);

ONNX_OPERATOR_KERNEL_EX(
    SequenceLength,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0),
    SequenceLength);

ONNX_OPERATOR_KERNEL_EX(
    ConcatFromSequence,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .Alias(0, 0),
    ConcatFromSequence);

ONNX_OPERATOR_KERNEL_EX(
    SequenceErase,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()})
        .Alias(0, 0),
    SequenceErase);

ONNX_OPERATOR_KERNEL_EX(
    SequenceInsert,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()})
        .Alias(0, 0),
    SequenceInsert);

} // namespace cuda
} // namespace onnxruntime


