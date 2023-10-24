// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/providers/js/js_kernel.h"
#include "core/providers/js/operators/cast.h"

namespace onnxruntime {
namespace js {

const std::vector<MLDataType>& CastOpTypeConstraints() {
  // currently support boolean, integer and float types that explicitly allowed in WGSL:
  // https://gpuweb.github.io/gpuweb/wgsl/#plain-types-section
  //
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<bool>()};
  return types;
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    6, 8,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    9, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    13, 18,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);
ONNX_OPERATOR_KERNEL_EX(
    Cast,
    kOnnxDomain,
    19,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);

}  // namespace js
}  // namespace onnxruntime
