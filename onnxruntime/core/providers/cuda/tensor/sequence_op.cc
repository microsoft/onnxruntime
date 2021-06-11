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
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceAt);

ONNX_OPERATOR_KERNEL_EX(
    SequenceConstruct,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SequenceConstruct);

ONNX_OPERATOR_KERNEL_EX(
    SequenceEmpty,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SequenceEmpty);

ONNX_OPERATOR_KERNEL_EX(
    SequenceLength,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    SequenceLength);

ONNX_OPERATOR_KERNEL_EX(
    ConcatFromSequence,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
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
                                 DataTypeImpl::GetTensorType<int64_t>()}),
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
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceInsert);

} // namespace cuda
} // namespace onnxruntime


