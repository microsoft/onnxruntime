// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qbias_gelu.h"

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/quantization/quantization.h"

namespace onnxruntime {
namespace contrib {

  #pragma warning(disable: 4189 4100)
namespace {
}  // namespace

//// This op is internal-only, so register outside of onnx:
//#define REGISTER_KERNEL_TYPED(T)                                  \
//  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
//      QBiasGelu,                                                  \
//      kMSDomain,                                                  \
//      1,                                                          \
//      T,                                                          \
//      kCpuExecutionProvider,                                      \
//      KernelDefBuilder()                                          \
//          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
//      QBiasGelu<T>);
//
//REGISTER_KERNEL_TYPED(float)
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QBiasGelu,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    QBiasGelu);

QBiasGelu::QBiasGelu(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
}

Status QBiasGelu::Compute(OpKernelContext* context) const {
  //ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));

  //bool is_signed_inputs = false;
  //ORT_RETURN_IF_ERROR(CheckQuantizedInputs(context, &is_signed_inputs));

  //if (is_signed_inputs) {
  //  return ComputeInternal<T, int8_t>(context, epsilon());
  //} else {
  //  return ComputeInternal<T, uint8_t>(context, epsilon());
  //}

  return Status::OK();
}

#pragma warning(default: 4189 4100)

}  // namespace contrib
}  // namespace onnxruntime
