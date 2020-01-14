// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul.h"

/*
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
*/

#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

namespace onnxruntime {
namespace acl {

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 9,
    float,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    1, 9,
    double,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    9, 9,
    int32_t,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    MatMul<int32_t>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    9, 9,
    uint32_t,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint32_t>()),
    MatMul<uint32_t>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    9, 9,
    int64_t,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    MatMul<int64_t>);

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    9, 9,
    uint64_t,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint64_t>()),
    MatMul<uint64_t>);

}  // namespace acl
}  // namespace onnxruntime
