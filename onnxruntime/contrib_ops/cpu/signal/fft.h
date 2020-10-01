// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

class Fft final : public OpKernel {
  int64_t signal_ndim_, normalized_, onesided_;
 public:

  explicit Fft(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class Ifft final : public OpKernel {
  int64_t signal_ndim_, normalized_, onesided_;
 public:
  explicit Ifft(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};


}  // namespace contrib
}  // namespace onnxruntime
