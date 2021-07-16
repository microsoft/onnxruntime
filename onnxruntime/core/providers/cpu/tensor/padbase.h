// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

enum class Mode : int {
  Constant = 0,
  Reflect,
  Edge
};

class PadBase {
 public:
  // Update the output_shape to make it consistent with numpy handling where there are one or more dimensions
  // in the input_shape with a value of zero.
  static Status HandleDimValueZero(const Mode& mode, const TensorShape& input_shape, TensorShape& output_shape);

 protected:
  PadBase(const OpKernelInfo& info) : value_(info.GetAttrOrDefault("value", 0.f)) {
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

    const auto& kernel_def = info.GetKernelDef();

    int start_ver, end_ver;
    kernel_def.SinceVersion(&start_ver, &end_ver);

    // kMSDomain contrib kernel AND OnnxDomain start version >= 11 => DynamicPad
    if (start_ver >= 11 || kernel_def.Domain() == kMSDomain) {
      is_dynamic_ = true;
    }

    if (!is_dynamic_) {
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
    }
  }

  ~PadBase() = default;

  Mode mode_{Mode::Constant};
  std::vector<int64_t> pads_;    // After construction, only >=0 values are in here
  std::vector<int64_t> slices_;  // All of the negative padding values are separated out into slices_
  const float value_;            // will always be float (when 'value' parsed from attribute - opset 10 and below)

  // flag used to differentiate the cases where some input values to the op are
  // to be obtained from (is_dynamic_ = false) attributes vs (is_dynamic_ = true) inputs
  bool is_dynamic_ = false;
};

}  // namespace onnxruntime
