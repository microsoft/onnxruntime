// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

template <typename T>
class Gelu final : public OpKernel {
 public:
  explicit Gelu(const OpKernelInfo& info) : OpKernel(info) {
    approximation_algorithm_ = info.GetAttrOrDefault<std::string>("approximate", "none");
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::string approximation_algorithm_;
};

}  // namespace onnxruntime
