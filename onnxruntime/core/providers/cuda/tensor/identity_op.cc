// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "identity_op.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    1, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    13, 13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

// From opset 14 onward the ONNX spec's type constraint is "V" which includes
// both Tensor and TensorSequence types.  In the plugin EP build TensorSeq is
// an incomplete type, so we register only the Tensor subset.
#ifdef BUILD_CUDA_EP_AS_PLUGIN
#define IDENTITY_V_TYPES DataTypeImpl::AllFixedSizeTensorTypes()
#define IDENTITY_V_TYPES_IRv9 DataTypeImpl::AllFixedSizeTensorTypes()
#else
#define IDENTITY_V_TYPES DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()
#define IDENTITY_V_TYPES_IRv9 DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9()
#endif

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    14, 18,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", IDENTITY_V_TYPES)
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    19, 20,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", IDENTITY_V_TYPES_IRv9)
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    21, 22,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", IDENTITY_V_TYPES_IRv9)
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    23, 24,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", IDENTITY_V_TYPES_IRv9)
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    Identity,
    kOnnxDomain,
    25,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", IDENTITY_V_TYPES_IRv9)
        .Alias(0, 0),
    IdentityOp<false>);

#undef IDENTITY_V_TYPES
#undef IDENTITY_V_TYPES_IRv9
}  // namespace cuda
}  // namespace onnxruntime
