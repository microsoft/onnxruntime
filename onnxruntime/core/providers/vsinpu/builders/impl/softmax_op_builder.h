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
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class SoftmaxOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    auto axis = helper.Get("axis", -1);
    auto input_defs = node->InputDefs();
    auto input_shape = vsi::npu::util::GetTensorShape(*input_defs[0]);
    int32_t rank = input_shape.NumDimensions();
    if (axis >= rank || axis < -rank) {
      LOGS_DEFAULT(ERROR) << "Axis is invalid in Softmax.";
      return false;
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Softmax Op.";
    NodeAttrHelper helper(node_unit.GetNode());
    int32_t def_val = node_unit.SinceVersion() < 13 ? 1 : -1;
    auto axis = helper.Get("axis", def_val);

    if (def_val == 1) {
      // In earlier opset version of softmax, input is coerced into 2D shape
      // Attribute "axis" is to describe the axis of the inputs when coerced to 2D but not take part in softmax computation
      const bool is_2d_shape = inputs[0]->GetShape().size() == 2 ? true : false;
      if (!is_2d_shape) {
        axis = HandleNegativeAxis(axis, inputs[0]->GetShape().size());
        auto it = inputs[0]->GetShape().end();
        uint32_t last_dim = std::accumulate(it - axis, it, 1, std::multiplies<uint32_t>());
        uint32_t first_dim = std::accumulate(inputs[0]->GetShape().begin(), it - axis, 1, std::multiplies<uint32_t>());
        auto reshaped_spec = inputs[0]->GetSpec().AsTransientSpec().SetShape(std::vector<uint32_t>{first_dim, last_dim});
        auto reshaped_input = graph_ep->GetGraph()->CreateTensor(reshaped_spec);
        auto reshaped_output = graph_ep->GetGraph()->CreateTensor(inputs[0]->GetSpec().AsTransientSpec());

        auto reshape_input_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(std::vector<uint32_t>{first_dim, last_dim});
        auto softmax_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Softmax>(1, 0);
        auto reshaped_output_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(inputs[0]->GetShape());

        (*reshape_input_op).BindInputs(inputs).BindOutput(reshaped_input);
        (*softmax_op).BindInput(reshaped_input).BindOutput(reshaped_output);
        (*reshaped_output_op).BindInput(reshaped_output).BindOutputs(outputs);

        graph_ep->GetOps().push_back(std::move(reshape_input_op));
        graph_ep->GetOps().push_back(std::move(softmax_op));
        graph_ep->GetOps().push_back(std::move(reshaped_output_op));
      } else {
        auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Softmax>(1, 0);
        (*op).BindInputs(inputs).BindOutputs(outputs);
        graph_ep->GetOps().push_back(std::move(op));
      }
    } else {
      axis = util::ReverseAxis(axis, inputs[0]->GetShape().size());
      auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Softmax>(1, static_cast<uint32_t>(axis));
      (*op).BindInputs(inputs).BindOutputs(outputs);
      graph_ep->GetOps().push_back(std::move(op));
    }
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
