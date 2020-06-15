// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/armnn/armnn_common.h"
#include "core/providers/armnn/nhwc/nhwc_ops.h"
#include "core/providers/armnn/armnn_fwd.h"

#include "armnn/ArmNN.hpp"

#include <thread>
#include <mutex>

namespace onnxruntime {
namespace armnn_ep {

const armnn::PermutationVector toNCHW = { 0, 2, 3, 1 };
const armnn::PermutationVector toNHWC = { 0, 3, 1, 2 };

armnn::TensorShape ARMNN_NCHW2NHWC(armnn::TensorShape shape) {

    return {shape[0], shape[2], shape[3], shape[1]};
}

template <typename T>
Status ReorderInput<T>::Compute(OpKernelContext* context) const {

  return ::onnxruntime::acl::ReorderInput<T>::Compute(context);

}

template <typename T>
Status ReorderOutput<T>::Compute(OpKernelContext* context) const {

  return ::onnxruntime::acl::ReorderOutput<T>::Compute(context);
}


ONNX_OPERATOR_TYPED_KERNEL_EX(
    ReorderInput,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderInput<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ReorderOutput,
    kMSNhwcDomain,
    1,
    float,
    kArmNNExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput<float>);

}  // namespace armnn_ep
}  // namespace onnxruntime
