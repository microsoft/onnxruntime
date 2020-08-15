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

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->template Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DropoutComputeImpl {
  void operator()(const cudaDeviceProp& prop,
                  const int64_t N,
                  const float ratio_data,
                  PhiloxGenerator& generator,
                  const Tensor& X,
                  Tensor& Y,
                  bool* mask_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.template Data<T>());
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.template MutableData<T>());

    DropoutKernelImpl<CudaT>(prop, N, ratio_data, generator, X_data, Y_data, mask_data);
  }
};

template <bool trainable_dropout>
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

template <bool trainable_dropout>
Status Dropout<trainable_dropout>::ComputeInternal(OpKernelContext* context) const {
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  const int64_t N = shape.Size();

  //Get Y_data
  auto Y = context->Output(0, shape);

  //Get mask_data
  auto mask = context->Output(1, shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    utils::MLTypeCallDispatcher<GetRatioDataImpl, float, MLFloat16, double> t_disp(ratio->GetElementType());
    t_disp.Invoke(ratio, ratio_data);
  }

  const Tensor* training_mode = context->Input<Tensor>(2);
  //Check for inference mode.
  if ((0 == ratio_data /*Backward compat with TrainableDropout*/) ||
      (!trainable_dropout && (training_mode == nullptr || *(training_mode->Data<bool>()) == false))) {
    const void* X_data = X->DataRaw();
    void* Y_data = Y->MutableDataRaw();
    if (Y_data != X_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(Y_data, X_data, X->SizeInBytes(), cudaMemcpyDeviceToDevice));
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

  utils::MLTypeCallDispatcher<DropoutComputeImpl, float, MLFloat16, double> t_disp(X->GetElementType());
  t_disp.Invoke(GetDeviceProp(), N, ratio_data, generator, *X, *Y, mask_data);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
