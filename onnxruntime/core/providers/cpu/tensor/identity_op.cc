// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/identity_op.h"
//#include "core/framework/op_lite.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Dropout,
    7, 9,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                                            DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>()}),
    IdentityOp<true>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Dropout,
    10,
    11,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                                            DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    IdentityOp<true>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Identity,
    1,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).Alias(0, 0),
    IdentityOp<false>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Identity,
    13,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).Alias(0, 0),
    IdentityOp<false>);

// Opset 14 supported sequence type
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Identity,
    14, 15,
    KernelDefBuilder().TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes()).Alias(0, 0),
    IdentityOp<false>);

// Opset 16 supported optional type
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Identity,
    16, 18,
    KernelDefBuilder().TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypes()).Alias(0, 0),
    IdentityOp<false>);

// Opset 19 supported float 8 types.
ONNX_CPU_OPERATOR_KERNEL(
    Identity,
    19,
    KernelDefBuilder().TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypesIRv9()).Alias(0, 0),
    IdentityOp<false>);

///////////////////////////////////////////// LITE OP ////////////////////////////////////////////////////

//onnxruntime::Status IdentityFooFn(const onnxruntime::lite::Tensor<float>& in, onnxruntime::lite::Tensor<float>& out) {
//  auto shape = in.Shape();
//  const float* raw_in = in.Data();
//  float* raw_out = out.Allocate(shape);
//  if (raw_out) {
//    for (int64_t i = 0; i < in.NumberOfElement(); ++i) {
//      raw_out[i] = raw_in[i];
//    }
//    return Status::OK();
//  } else {
//    return ORT_MAKE_STATUS(StatusCategory::ONNXRUNTIME, StatusCode::RUNTIME_EXCEPTION);
//  }
//}
//
//ONNX_OP_BY_FN(
//    IdentityFoo,
//    kOnnxDomain,
//    1,
//    kCpuExecutionProvider,
//    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>()}),
//    IdentityFooFn);

#define ORT_NATIVE_OP
#include "core/session/onnxruntime_lite_custom_op.h"

::onnxruntime::Status IdentityFooFn(const onnxruntime::lite::Tensor<float>& in, onnxruntime::lite::Tensor<float>& out) {
   auto shape = in.Shape();
   const float* raw_in = in.Data();
   float* raw_out = out.Allocate(shape);
   if (raw_out) {
     for (int64_t i = 0; i < in.NumberOfElement(); ++i) {
       raw_out[i] = raw_in[i];
     }
     return Status::OK();
   } else {
     return ORT_MAKE_STATUS(StatusCategory::ONNXRUNTIME, StatusCode::RUNTIME_EXCEPTION);
   }
 }

}  // namespace onnxruntime
