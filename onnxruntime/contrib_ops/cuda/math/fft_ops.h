// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "cufft.h"
#include "cufftXt.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class FFTBase : public ::onnxruntime::cuda::CudaKernel {
 public:
  FFTBase(const OpKernelInfo info) : ::onnxruntime::cuda::CudaKernel{info}, normalized_{0}, onesided_{1} {
    ORT_ENFORCE((info.GetAttr("signal_ndim", &signal_ndim_)).IsOK(),
                "Attribute signal_ndim is missing in Node ", info.node().Name());
    ORT_ENFORCE(signal_ndim_ >= 1 && signal_ndim_ <= 3,
                "Expected signal_ndim to be 1, 2, or 3, but got signal_ndim=", signal_ndim_);
    info.GetAttr("normalized", &normalized_);
    info.GetAttr("onesided", &onesided_);
  }

 protected:
  int64_t signal_ndim_, normalized_, onesided_;
  Status DoFFT(OpKernelContext* context, const Tensor* X, bool complex_input, bool complex_output, bool inverse) const;
};

template <typename T>
class Rfft final : public FFTBase<T> {
 public:
  Rfft(const OpKernelInfo info) : FFTBase{info} {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Irfft final : public FFTBase<T> {
 public:
  Irfft(const OpKernelInfo info) : FFTBase{info} {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
