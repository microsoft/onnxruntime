// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <bool is_dropout>
class IdentityOp final : public CudaKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    auto X_ml_type = context->InputType(0);
    if (X_ml_type->IsTensorType()) {
      const Tensor* X = context->Input<Tensor>(0);
      if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
      const TensorShape& shape = X->Shape();
      Tensor* Y = context->Output(0, shape);
      auto X_type = X->DataType();

      const void* source = X->DataRaw(X_type);
      void* target = Y->MutableDataRaw(X_type);
      //If source and target pointers are not equal, we need to copy the data.
      if (target != source) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, X->Shape().Size() * X->DataType()->Size(), cudaMemcpyDeviceToDevice, Stream()));
      }

      if (is_dropout) {
        Tensor* mask = context->Output(1, shape);
        // a 'nullptr' returned would make it an unused optional output
        if (mask != nullptr) {
          // Opset 7 differs with Opset 10 in that the type of the 'mask'
          // output is tied with the type of the input in Opset 7 whereas
          // the type of 'mask' in Opset 10 is 'bool' always
          // so we have a common solution
          void* mask_data = mask->MutableDataRaw();
          // In 'test'/'inference' mode, there are no input values dropped out
          // so fill the buffer with 0/false
          CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mask_data, 0, mask->SizeInBytes(), Stream()));
        }
      }
    } else if (X_ml_type->IsTensorSequenceType()) {
      const TensorSeq* X = context->Input<TensorSeq>(0);
      ORT_ENFORCE(X != nullptr);
      TensorSeq* Y = context->Output<TensorSeq>(0);
      ORT_ENFORCE(Y->Size() == 0, "output tensor sequence not empty");
      auto X_type = X->DataType();
      Y->SetType(X_type);
      AllocatorPtr alloc;
      auto status = context->GetTempSpaceAllocator(&alloc);
      if (!status.IsOK()) {
        ORT_THROW("Unable to get an allocator");
      }
      for (size_t i = 0; i < X->Size(); ++i) {
        const Tensor& source_tensor = X->Get(i);
        auto target_tensor = Tensor::Create(X_type, source_tensor.Shape(), alloc);
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                             source_tensor.DataRaw(),
                                             source_tensor.SizeInBytes(),
                                             cudaMemcpyDeviceToDevice, Stream()));
        Y->Add(std::move(*target_tensor));
      }
    } else {
      ORT_THROW("Unsupported input type");
    }
    return Status::OK();
  }
};

}  // namespace cuda
}  // namespace onnxruntime
