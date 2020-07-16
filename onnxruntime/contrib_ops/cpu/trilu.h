// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

class Trilu final : public OpKernel {
 public:
  explicit Trilu(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("upper", &upper_).IsOK());
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
   int64_t upper_;
   template<typename T>
   struct ComputeImpl;
};

}  // namespace contrib
}  // namespace onnxruntime
