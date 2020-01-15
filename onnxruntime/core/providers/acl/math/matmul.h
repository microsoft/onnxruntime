// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace acl {

template <typename T>
class MatMul final : public onnxruntime::MatMul<T> {
 public:
  MatMul(const OpKernelInfo& info)
      : onnxruntime::MatMul<T>(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* left_X = ctx->Input<Tensor>(0);
    const Tensor* right_X = ctx->Input<Tensor>(1);

    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape()));

    Tensor* Y = ctx->Output(0, helper.OutputShape());

    LOGS_DEFAULT(VERBOSE) << "MatMul ACL:" << std::endl;
    if (left_X) LOGS_DEFAULT(VERBOSE) << "left_X " << left_X->Shape().ToString().c_str() << std::endl;
    if (right_X) LOGS_DEFAULT(VERBOSE) << "right_X " << right_X->Shape().ToString().c_str() << std::endl;
    if (Y) LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << std::endl;

    return onnxruntime::MatMul<T>::Compute(ctx);
  }
};

}  // namespace acl
}  // namespace onnxruntime
