// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class DynamicPadBase {
 protected:
  DynamicPadBase(const OpKernelInfo& info) {
    std::string mode;
    if (info.GetAttr("mode", &mode).IsOK()) {
      if (mode == "constant")
        mode_ = Mode::Constant;
      else if (mode == "reflect")
        mode_ = Mode::Reflect;
      else if (mode == "edge")
        mode_ = Mode::Edge;
      else
        ORT_THROW("Invalid 'mode' attribute value");
    }
  }

  ~DynamicPadBase() {}

  enum class Mode : int {
    Constant = 0,
    Reflect,
    Edge
  };
  Mode mode_{Mode::Constant};
};

template <typename T>
struct DynamicPad final : public OpKernel, public DynamicPadBase {
  DynamicPad(const OpKernelInfo& info) : OpKernel(info), DynamicPadBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}
}  // namespace onnxruntime