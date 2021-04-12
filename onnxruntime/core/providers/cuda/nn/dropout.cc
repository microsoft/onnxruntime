// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

namespace onnxruntime {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define ALL_IEEE_FLOAT_TENSOR_TYPES           \
  { DataTypeImpl::GetTensorType<float>(),     \
    DataTypeImpl::GetTensorType<double>(),    \
    DataTypeImpl::GetTensorType<MLFloat16>(), \
    DataTypeImpl::GetTensorType<BFloat16>() }
#else
#define ALL_IEEE_FLOAT_TENSOR_TYPES DataTypeImpl::AllIEEEFloatTensorTypes()
#endif

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    12, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    Dropout);

ONNX_OPERATOR_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", ALL_IEEE_FLOAT_TENSOR_TYPES)
        .TypeConstraint("T1", ALL_IEEE_FLOAT_TENSOR_TYPES)
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    Dropout);

}  // namespace cuda
}  // namespace onnxruntime
