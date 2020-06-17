// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/nn/dropout_impl.h"
#include "core/providers/cuda/nn/dropout.h"
#include "core/providers/common.h"
#include "core/framework/random_seed.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2, bool trainable_dropout>
class Dropout final : public CudaKernel {
 public:
  Dropout(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = onnxruntime::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

template <typename T1, typename T2, bool trainable_dropout>
Status Dropout<T1, T2, trainable_dropout>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T1>::MappedType CudaT;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T1>());
  const int64_t N = shape.Size();

  //Get Y_data
  auto Y = context->Output(0, shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T1>());

  //Get mask_data
  auto mask = context->Output(1, shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  //Get the ratio_data
  float ratio_data;
  auto ratio = context->Input<Tensor>(1);

  static_assert(std::is_same<T2, MLFloat16>::value || std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                "T2 must be float16 or float or double");

  if (ratio) {
    ratio_data = static_cast<float>(*(ratio->template Data<T2>()));
  } else {
    ratio_data = default_ratio_;
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  const Tensor* training_mode = context->Input<Tensor>(2);
  //Check for inference mode.
  if ((0 == ratio_data /*Backward compat with TrainableDropout*/) ||
      (!trainable_dropout && (training_mode == nullptr || *(training_mode->Data<bool>()) == false))) {
    if (Y_data != X_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(Y_data, X_data, N * sizeof(T1), cudaMemcpyDeviceToDevice));
    }

    // If mask is requested, return all 1s.
    if (mask != nullptr) {
      ORT_ENFORCE(cudaMemset(mask->MutableData<bool>(), true, N * sizeof(bool)) == cudaSuccess);
    }

    return Status::OK();
  }

  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
  DropoutKernelImpl(GetDeviceProp(), N, ratio_data, generator, X_data, Y_data, mask_data);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
