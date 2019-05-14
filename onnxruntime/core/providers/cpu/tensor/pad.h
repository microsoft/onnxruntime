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

class PadBase {
 protected:
  PadBase(const OpKernelInfo& info, bool dynamic = false) : value_(info.GetAttrOrDefault("value", 0.f)) {
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

      // Separate out any negative pads_ into the slices_ array
      slices_.resize(pads_.size(), 0);
      for (size_t index = 0; index < pads_.size(); index++) {
        if (pads_[index] < 0) {
          slices_[index] = pads_[index];
          pads_[index] = 0;
        }
      }

      ;  // Value is optional and initialized to 0 by default
    }
  }

  ~PadBase() {}

  Mode mode_{Mode::Constant};
  std::vector<int64_t> pads_;    // After construction, only >=0 values are in here
  std::vector<int64_t> slices_;  // All of the negative padding values are separated out into slices_
  const float value_;            // will always be float (when 'value' parsed from attribute - opset 10 and below)
};

template <typename T>
struct Pad final : public OpKernel, public PadBase {
  Pad(const OpKernelInfo& info) : OpKernel(info), PadBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
Status PadCpuImpl(OpKernelContext* ctx,
                  const std::vector<int64_t>& pads,
                  const std::vector<int64_t>& slices,
                  const Mode& mode,
                  T value);

}  // namespace onnxruntime
