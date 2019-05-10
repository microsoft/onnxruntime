// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

enum class Mode : int {
  Constant = 0,
  Reflect,
  Edge
};

template <class T>
class PadBase {
 protected:
  PadBase(const OpKernelInfo& info, bool dynamic = false) : value_(info.GetAttrOrDefault("value", static_cast<T>(0))) {
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

    if (!dynamic) {
      if (!info.GetAttrs("pads", pads_).IsOK())
        ORT_THROW("Invalid 'pads' attribute value");

      ;  // Value is optional and initialized to 0 by default
    }
  }

  ~PadBase() {}

  Mode mode_{Mode::Constant};
  std::vector<int64_t> pads_;
  const T value_;
};

template <typename T>
struct Pad final : public OpKernel, public PadBase<float> {
  Pad(const OpKernelInfo& info) : OpKernel(info), PadBase<float>(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
Status PadImpl(OpKernelContext* ctx,
               const std::vector<int64_t>& raw_pads,
               const Mode& mode,
               T value);

}  // namespace onnxruntime
