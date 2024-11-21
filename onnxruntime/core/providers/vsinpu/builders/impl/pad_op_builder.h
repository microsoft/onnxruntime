/****************************************************************************
 *
 *    Copyright (c) 2024 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#pragma once
#include <memory>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include "core/optimizer/initializer.h"
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {

typedef tim::vx::ops::PadV2::pad_mode_type PadMode;

class PadOpBuilder : public BaseOpBuilder {
 public:
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 11; }
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    const auto mode = helper.Get("mode", "constant");
    auto input_defs = node->InputDefs();
    size_t num_inputs = input_defs.size();
    auto input_shape = vsi::npu::util::GetTensorShape(*input_defs[0]);
    int32_t rank = input_shape.NumDimensions();
    const auto& initializers = graph_viewer.GetAllInitializedTensors();

    if (mode == "wrap") {
      LOGS_DEFAULT(WARNING) << "`wrap` mode Pad is not currently supported for now.";
      return false;
    }
    if (mode == "constant") {
      if (num_inputs > 2 && input_defs[2]->Exists()) {
        // only support if `constant_value` input is a constant initializer
        if (!Contains(initializers, input_defs[2]->Name())) {
          LOGS_DEFAULT(WARNING) << "constant_value must be a constant initializer.";
          return false;
        }
      }
    }
    // only support if `pads` input is known and does not contain negative values
    {
      const auto* pads_initializer = graph_viewer.GetConstantInitializer(input_defs[1]->Name());
      if (!pads_initializer) {
        LOGS_DEFAULT(WARNING) << "pads must be a constant initializer";
        return false;
      }

      Initializer unpacked_tensor(*pads_initializer);
      auto tensor_data = unpacked_tensor.DataAsSpan<int64_t>();
      for (size_t i = 0; i < unpacked_tensor.size(); i++) {
        if (tensor_data[i] < 0) {
          LOGS_DEFAULT(WARNING) << "Negative pad value is not supported: pads["
                                << i << "] = " << tensor_data[i];
          return false;
        }
      }
    }
    return true;
  }

  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    for (size_t i = 0; i < node_unit.Inputs().size(); ++i) {
      const auto& iodef = node_unit.Inputs()[i];
      if (0 == i) {
        if (!util::IsTypeSupported(&iodef.node_arg) ||
            (*iodef.node_arg.Type() == "tensor(int64)") ||
            (*iodef.node_arg.Type() == "tensor(bool)")) {
          LOGS_DEFAULT(WARNING) << "Unspport tensor data type:" << *iodef.node_arg.Type();
          return false;
        }
      } else if (1 == i) {
        if (!Contains(initializers, iodef.node_arg.Name())) {
          LOGS_DEFAULT(WARNING) << "pads must be a constant initializer.";
          return false;
        }
      } else if (2 == i) {
        if (iodef.node_arg.Exists() && !Contains(initializers, iodef.node_arg.Name())) {
          LOGS_DEFAULT(WARNING) << "constant_value must be a constant initializer.";
          return false;
        }
      } else if (i == 3) {
        if (!Contains(initializers, iodef.node_arg.Name())) {
          LOGS_DEFAULT(WARNING) << "axes must be a constant initializer..";
          return false;
        }
      }
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Pad Op.";
    NodeAttrHelper helper(node_unit);
    const auto mode = helper.Get("mode", "constant");
    auto input_defs = node_unit.Inputs();
    PadMode pad_mode = PadMode::PAD_MODE_CONSTANT;
    float const_val = 0.0f;
    std::vector<int64_t> axes_tensor_data;
    int32_t input_rank = inputs[0]->GetShape().size();

    if (mode == "constant") {
      pad_mode = PadMode::PAD_MODE_CONSTANT;
    } else if (mode == "reflect") {
      pad_mode = PadMode::PAD_MODE_REFLECT;
    } else if (mode == "edge") {
      pad_mode = PadMode::PAD_MODE_EDGE;
    } else {
      LOGS_DEFAULT(WARNING) << "`wrap` mode Pad is not currently supported for now.";
      return false;
    }

    // `pads` input
    std::vector<int64_t> onnx_pads(inputs[1]->GetSpec().GetElementNum());
    inputs[1]->CopyDataFromTensor(onnx_pads.data());

    // `constant_value` input
    if (inputs.size() > 2 && pad_mode == PadMode::PAD_MODE_CONSTANT) {
      if (input_defs[2].node_arg.Exists()) {
        inputs[2]->CopyDataFromTensor(&const_val);
      }
    }
    // `axes` input
    if (inputs.size() > 3) {
      // optional input axes is provided, use axes initializer data
      std::vector<int64_t> axes_tensor(inputs[3]->GetSpec().GetElementNum());
      inputs[3]->CopyDataFromTensor(axes_tensor.data());
      std::transform(
          axes_tensor.begin(), axes_tensor.end(), std::back_inserter(axes_tensor_data),
          [input_rank](int64_t axis) { return HandleNegativeAxis(axis, input_rank); });
    } else {
      // if not provided, make a default axes as [0, 1, ..., input_rank - 1]
      std::vector<int64_t> default_axes(input_rank);
      std::iota(std::begin(default_axes), std::end(default_axes), 0);
      axes_tensor_data = std::move(default_axes);
    }

    int64_t num_axes = axes_tensor_data.size();
    std::vector<uint32_t> front_size(input_rank, 0);
    std::vector<uint32_t> back_size(input_rank, 0);

    int64_t axes_index = 0;
    for (int64_t axes : axes_tensor_data) {
      front_size[axes] = onnx_pads[axes_index];
      back_size[axes] = onnx_pads[axes_index + num_axes];
      axes_index++;
    }

    std::reverse(front_size.begin(), front_size.end());
    std::reverse(back_size.begin(), back_size.end());

    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::PadV2>(
        front_size, back_size, const_val, pad_mode);
    op->BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
