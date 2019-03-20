// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "core/common/common.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "core/framework/op_kernel.h"

namespace onnxruntime {

// Helper to copy from an input Tensor to the specified output index.
// In a struct just to provide some context to the method name.
struct IdentityOpHelper {
  static void CopyInputToOutput(OpKernelContext* context, const Tensor& input, int output_index);
};

template <bool is_dropout>
class IdentityOp final : public OpKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr);

    IdentityOpHelper::CopyInputToOutput(context, *X, 0);

    if (is_dropout) {
      context->Output(1, std::vector<int64_t>());
    }

    return Status::OK();
  }
};

}  //namespace onnxruntime
