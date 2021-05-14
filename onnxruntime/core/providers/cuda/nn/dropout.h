// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/nn/dropout_impl.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define ALL_IEEE_FLOAT_TENSOR_TYPES           \
  { DataTypeImpl::GetTensorType<float>(),     \
    DataTypeImpl::GetTensorType<double>(),    \
    DataTypeImpl::GetTensorType<MLFloat16>(), \
    DataTypeImpl::GetTensorType<BFloat16>() }
#define ALL_IEEE_FLOAT_DATA_TYPES float, MLFloat16, double, BFloat16
#else
#define ALL_IEEE_FLOAT_TENSOR_TYPES DataTypeImpl::AllIEEEFloatTensorTypes()
#define ALL_IEEE_FLOAT_DATA_TYPES float, MLFloat16, double
#endif

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
                  cudaStream_t stream,
                  const int64_t N,
                  const float ratio_data,
                  PhiloxGenerator& generator,
                  const Tensor& X,
                  Tensor& Y,
                  bool* mask_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.template Data<T>());
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.template MutableData<T>());

    DropoutKernelImpl<CudaT>(prop, stream, N, ratio_data, generator, X_data, Y_data, mask_data);
  }
};

class Dropout final : public CudaKernel {
 public:
  Dropout(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace onnxruntime
