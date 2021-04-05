// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {

class Trilu final : public OpKernel {
 public:
  explicit Trilu(const OpKernelInfo& info) : OpKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("upper", &temp).IsOK());
    upper_ = temp != 0;
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
   bool upper_;
};

}  // namespace onnxruntime
