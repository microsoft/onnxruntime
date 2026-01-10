// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {

class GatherBase {
 public:
  struct Prepare {
    const Tensor* input_tensor;
    const Tensor* indices_tensor;
    Tensor* output_tensor;
    int64_t axis;
  };

  Status PrepareForCompute(OpKernelContext* context, Prepare& p) const;

 protected:
  GatherBase(const OpKernelInfo& info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }

 private:
  int64_t axis_;
};

}  // namespace onnxruntime
