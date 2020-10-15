// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

class DFT final : public OpKernel {
  int64_t signal_ndim_ = 1;
 public:

  explicit DFT(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class IDFT final : public OpKernel {
  int64_t signal_ndim_ = 1;
 public:
  explicit IDFT(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class STFT final : public OpKernel {
 public:
  explicit STFT(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class ISTFT final : public OpKernel {
 public:
  explicit ISTFT(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
