// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

namespace onnxruntime {
namespace contrib {

class DFT final : public OpKernel {
  bool is_onesided_ = true;
 public:
  explicit DFT(const OpKernelInfo& info) : OpKernel(info) {
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 0);
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class IDFT final : public OpKernel {
 public:
  explicit IDFT(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override;
};

class STFT final : public OpKernel {
  bool is_onesided_ = true;
 public:
  explicit STFT(const OpKernelInfo& info) : OpKernel(info) {
    is_onesided_ = info.GetAttrOrDefault<int64_t>("onesided", 1);
  }
  Status Compute(OpKernelContext* ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif