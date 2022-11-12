// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

template <bool is_dropout>
class IdentityOp final : public CannKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : CannKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    auto X_ml_type = context->InputType(0);
    if (X_ml_type->IsTensorType()) {
      const Tensor* X = context->Input<Tensor>(0);
      if (nullptr == X) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "IdentityOp cann: input count mismatch.");
      }
      const TensorShape& shape = X->Shape();
      Tensor* Y = context->Output(0, shape);
      if (nullptr == Y) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "IdentityOp cann: failed to allocate output tensor.");
      }
      auto X_type = X->DataType();

      const void* source = X->DataRaw(X_type);
      void* target = Y->MutableDataRaw(X_type);
      if (target != source) {
        CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(target, Y->SizeInBytes(), source,
                                              X->Shape().Size() * X->DataType()->Size(),
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, Stream()));
      }

      if (is_dropout) {
        Tensor* mask = context->Output(1, shape);
        if (mask != nullptr) {
          void* mask_data = mask->MutableDataRaw();
          CANN_RETURN_IF_ERROR(aclrtMemsetAsync(mask_data, mask->SizeInBytes(), 0, mask->SizeInBytes(), Stream()));
        }
      }
    } else if (X_ml_type->IsTensorSequenceType()) {
      const TensorSeq* X = context->Input<TensorSeq>(0);
      ORT_ENFORCE(X != nullptr, "IdentityOp cann: input tensor is missing.");
      TensorSeq* Y = context->Output<TensorSeq>(0);
      ORT_ENFORCE(Y != nullptr, "IdentityOp cann: failed to allocate output tensor sequence.");
      if (X == Y) {
        return Status::OK();
      }
      auto X_type = X->DataType();
      Y->SetType(X_type);
      AllocatorPtr alloc;
      auto status = context->GetTempSpaceAllocator(&alloc);
      if (!status.IsOK()) {
        return Status(common::ONNXRUNTIME, common::FAIL,
                      "IdentityOp cann: unable to get an allocator.");
      }
      auto X_size = X->Size();
      for (size_t i = 0; i < X_size; ++i) {
        const Tensor& source_tensor = X->Get(i);
        std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                               source_tensor.Shape(), alloc);
        CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(target_tensor->MutableDataRaw(),
                                              target_tensor->SizeInBytes(),
                                              source_tensor.DataRaw(),
                                              source_tensor.SizeInBytes(),
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, Stream()));
        Y->Add(std::move(*target_tensor));
      }
    } else {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "IdentityOp cann: unsupported input type.");
    }
    return Status::OK();
  }
};

}  // namespace cann
}  // namespace onnxruntime
