// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
namespace cuda {

template <bool is_dropout>
class IdentityOp final : public CudaKernel {
 public:
  IdentityOp(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    auto X_ml_type = context->InputType(0);
    if (DataTypeImpl::GetType<Tensor>() == X_ml_type) {
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
    } else if (DataTypeImpl::GetType<TensorSeq>() == X_ml_type) {
      const auto* X = context->Input<TensorSeq>(0);
      ORT_ENFORCE(X != nullptr);
      TensorSeq* Y = context->Output<TensorSeq>(0);
      auto X_type = X->DataType();
      Y->SetType(X_type);
      AllocatorPtr alloc;
      auto status = context->GetTempSpaceAllocator(&alloc);
      if (!status.IsOK()) {
        ORT_THROW("Unable to get an allocator");
      }
      std::vector<Tensor> tensors;
      for (auto iter = X->begin(); iter != X->end(); ++iter) {
        Tensor tensor(X_type, onnxruntime::TensorShape(iter->Shape()), alloc);
        size_t bytes = iter->SizeInBytes();
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(tensor.MutableDataRaw(), iter->DataRaw(), bytes, cudaMemcpyDeviceToDevice, Stream()));
        tensors.push_back(std::move(tensor));
      }
      Y->SetElements(std::move(tensors));
    } else {
      ORT_THROW("Unsupported input type");
    }
    return Status::OK();
  }
};

}  // namespace cuda
}  // namespace onnxruntime
