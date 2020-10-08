// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

class Dft final : public OpKernel {
  int64_t signal_ndim_, normalized_, onesided_;
 public:

  explicit Dft(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class Idft final : public OpKernel {
  int64_t signal_ndim_, normalized_, onesided_;
 public:
  explicit Idft(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};


}  // namespace contrib
}  // namespace onnxruntime
