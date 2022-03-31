// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/concat_impl.h"

namespace onnxruntime {
namespace cuda {

class InPlaceToSequence final : public CudaKernel {
 public:
  InPlaceToSequence(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    auto num_inputs = Node().InputArgCount().front();
    ORT_ENFORCE(num_inputs >= 1, "Must have 1 or more inputs");

    MLDataType first_dtype = context->Input<Tensor>(0)->DataType();

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "InPlaceToSequence GPU: Unable to get an allocator.");

    TensorSeq* Y = context->Output<TensorSeq>(0);
    Y->SetType(first_dtype);
    Y->Reserve(num_inputs);

    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
      auto* source_tensor = const_cast<Tensor*>(context->Input<Tensor>(input_idx));
      Y->Add(std::move(*source_tensor));  // Add will check for type consistency
    }

    return Status::OK();
  }
};

}  // namespace cuda
}  // namespace onnxruntime
