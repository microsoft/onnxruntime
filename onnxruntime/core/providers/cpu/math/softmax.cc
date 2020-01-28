// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_config.h"
//Ignore a wired warning in gcc 7.4.0. The latest gcc doesn't generate this warning
#ifdef __GNUC__
#ifdef HAS_MAYBE_UNINITIALIZED
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
#include "core/providers/cpu/math/softmax.h"

#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/eigen_common_wrapper.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Softmax, 1, 10, float,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                         Softmax<float, false>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(Softmax, 11, float, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                               Softmax<float, false>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(LogSoftmax, 1, 10, float,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                         Softmax<float, true>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(LogSoftmax, 11, float, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                               Softmax<float, true>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(Softmax, 1, 10, double,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                                         Softmax<double, false>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(Softmax, 11, double, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                               Softmax<double, false>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(LogSoftmax, 1, 10, double,
                                         KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                                         Softmax<double, true>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(LogSoftmax, 11, double, KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                               Softmax<double, true>);
}  // namespace onnxruntime
