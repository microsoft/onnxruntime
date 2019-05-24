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

template <bool is_dropout>
class IdentityOp final : public OpKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr);
    const TensorShape& shape = X->Shape();
    Tensor* Y = context->Output(0, shape);
    auto X_type = X->DataType();

    const void* source = X->DataRaw(X_type);
    void* target = Y->MutableDataRaw(X_type);
    //If source and target pointers are not equal, we need to copy the data.
    if (target != source) {
      if (X_type != DataTypeImpl::GetType<std::string>()) {
        memcpy(target, source, shape.Size() * X_type->Size());
      } else {
        // handle std::string
        const auto* src = X->template Data<std::string>();
        auto* dst = Y->template MutableData<std::string>();
        std::copy(src, src + shape.Size(), dst);
      }
    }

    if (is_dropout) {
      context->Output(1, std::vector<int64_t>());
    }

    return Status::OK();
  }
};

}  //namespace onnxruntime
