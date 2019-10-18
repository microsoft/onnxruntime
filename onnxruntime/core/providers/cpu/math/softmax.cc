// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cpu/math/softmax.h"

#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/eigen_common_wrapper.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(Softmax, 1, 10,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Softmax<float, false>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_KERNEL(Softmax, 11, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Softmax<float, false>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(LogSoftmax, 1, 10,
                                   KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                   Softmax<float, true>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_KERNEL(LogSoftmax, 11, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                         Softmax<float, true>);

}  // namespace onnxruntime
