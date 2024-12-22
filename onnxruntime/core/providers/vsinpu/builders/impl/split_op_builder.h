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

class SplitOpBuilder : public BaseOpBuilder {
 public:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    auto axis = helper.Get("axis", 0);
    auto input_defs = node->InputDefs();
    size_t num_inputs = input_defs.size();
    size_t num_outputs = node->OutputDefs().size();
    auto input_shape = vsi::npu::util::GetTensorShape(*input_defs[0]);
    int32_t rank = input_shape.NumDimensions();
    std::vector<int64_t> splits_list;
    bool split_provided = false;
    if (axis >= rank || axis < -rank) {
      LOGS_DEFAULT(WARNING) << "Axis is invalid in Split. Axis(" << axis
                            << ") is out of rank[" << -rank << "," << rank - 1 << "]";
      return false;
    }
    axis = HandleNegativeAxis(axis, rank);
    const auto split_dims_at_axis = input_shape.GetDims()[axis];
    if (num_inputs > 1 && input_defs[1]->Exists()) {
      // if optional input `split` is provided
      const auto* splits = graph_viewer.GetConstantInitializer(input_defs[1]->Name());
      if (!splits) {
        LOGS_DEFAULT(WARNING) << "Optional input 'split' must be a constant initializer if provided.";
        return false;
      }
      Initializer unpacked_tensor(*splits);
      auto split_sizes_ = unpacked_tensor.DataAsSpan<int64_t>();
      splits_list.assign(split_sizes_.begin(), split_sizes_.end());
      split_provided = true;
    }
    if (num_inputs == 1) {
      // opset1,2,11 split as attribute
      if (helper.HasAttr("split")) {
        auto split_sizes_ = *helper.GetInt64s("split");
        splits_list.assign(split_sizes_.begin(), split_sizes_.end());
        split_provided = true;
      } else if (node->SinceVersion() >= 18) {
        const auto outputs_count = helper.GetInt64("num_outputs");
        if (!outputs_count.has_value()) {
          LOGS_DEFAULT(WARNING) << "No 'num_outputs' provided. For split 18+, num_outputs is a required attribute.";
          return false;
        }
        if (outputs_count.value() != static_cast<int32_t>(num_outputs) ||
            outputs_count.value() > split_dims_at_axis) {
          LOGS_DEFAULT(WARNING) << "Invalid num_outputs provided.\n. The value should be smaller or equal to the size "
                                   "of dimension being split. num_outputs: "
                                << outputs_count.value();
          return false;
        }
      }
    }
    if (!split_provided) {
      // populate split sizes based on num_outputs so existing code can be utilized
      int32_t size = narrow<int32_t>(std::ceil(float(split_dims_at_axis) / num_outputs));
      int32_t remainder = split_dims_at_axis % size;
      std::vector<int64_t> split_sizes_(num_outputs, size);
      if (remainder) {
        split_sizes_.back() = remainder;
      }
      splits_list.assign(split_sizes_.begin(), split_sizes_.end());
    }

    uint32_t sum_of_splits = std::accumulate(splits_list.begin(), splits_list.end(), SafeInt<uint32_t>(0));
    if (sum_of_splits != split_dims_at_axis) {
      LOGS_DEFAULT(WARNING) << "Sum of the 'split' input values must equal to the dim value at 'axis' specified. "
                            << "dim value at 'axis' specified: "
                            << split_dims_at_axis
                            << ", sum of 'split' input values: "
                            << sum_of_splits;
      return false;
    }
    if (!std::all_of(splits_list.begin(), splits_list.end(), [](int64_t value) { return value >= 0; })) {
      LOGS_DEFAULT(WARNING) << "Invalid value in 'split' attribute. All values must be > 0";
      return false;
    }
    auto average_split = sum_of_splits / num_outputs;
    if (!std::all_of(splits_list.begin(), splits_list.end(), [average_split](int64_t value) { return value == average_split; })) {
      // TO DO, remove this check after driver supports it.
      LOGS_DEFAULT(WARNING) << "Uneven splits are not currently supported for now.";
      return false;
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
          LOGS_DEFAULT(WARNING) << "Unsupport tensor data type:" << *iodef.node_arg.Type();
          return false;
        }
      } else if (!Contains(initializers, iodef.node_arg.Name())) {
        LOGS_DEFAULT(WARNING) << "Optional input 'split' must be a constant initializer if provided.";
        return false;
      }
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Split Op.";
    NodeAttrHelper helper(node_unit);
    auto axis = helper.Get("axis", 0);
    axis = util::ReverseAxis(axis, inputs[0]->GetShape().size());
    const auto split_dims_at_axis = inputs[0]->GetShape()[axis];
    auto num_outputs = outputs.size();
    // transform splite vector to timvx slice
    std::vector<int64_t> onnx_split;
    if (inputs.size() > 1) {
      std::vector<int64_t> split_sizes_(inputs[1]->GetSpec().GetElementNum());
      inputs[1]->CopyDataFromTensor(split_sizes_.data());
      onnx_split.assign(split_sizes_.begin(), split_sizes_.end());
    }
    if (inputs.size() == 1) {
      if (helper.HasAttr("split")) {
        auto split_sizes_ = *helper.GetInt64s("split");
        onnx_split.assign(split_sizes_.begin(), split_sizes_.end());
      }
      if (node_unit.SinceVersion() >= 18 || !helper.HasAttr("split")) {
        // populate split sizes based on num_outputs so existing code can be utilized
        int32_t size = narrow<int32_t>(std::ceil(float(split_dims_at_axis) / num_outputs));
        int32_t remainder = split_dims_at_axis % size;
        std::vector<int64_t> split_sizes_(num_outputs, size);
        if (remainder) {
          split_sizes_.back() = remainder;
        }
        onnx_split.assign(split_sizes_.begin(), split_sizes_.end());
      }
    }
    std::vector<uint32_t> slices(onnx_split.begin(), onnx_split.end());
    std::reverse(slices.begin(), slices.end());

    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Split>(
        axis, slices);
    op->BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
