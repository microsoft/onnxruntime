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
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class ReduceMeanOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    if (*input_defs[0]->Type() == "tensor(int32)") {
      LOGS_DEFAULT(WARNING) << "Not support int32 reduce mean yet.";
      return false;
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                   const NodeUnit& node_unit) override {
    LOGS_DEFAULT(INFO) << "Creating ReduceMean Op.";

    NodeAttrHelper helper(node_unit.GetNode());
    std::vector<int64_t> def_axes;
    auto input_shape_size = inputs[0]->GetShape().size();

    if (node_unit.SinceVersion() < 18 && helper.HasAttr("axes")) {
        def_axes = helper.Get("axes", def_axes);
    } else if (inputs.size() > 1) {
        def_axes.resize(inputs[1]->GetSpec().GetElementNum());
        inputs[1]->CopyDataFromTensor(def_axes.data());
    } else {
        for (int64_t i = 0; i < input_shape_size; ++i) {
            def_axes.push_back(i);
        }
    }

    std::vector<int32_t> axes(def_axes.begin(), def_axes.end());
    axes = util::ReverseAxis(axes, input_shape_size);

    if (helper.HasAttr("noop_with_empty_axes") && inputs.size() == 1 && helper.Get("noop_with_empty_axes", 0) == 1) {
        outputs[0] = inputs[0];
        return true;
    }

    bool keepdims = helper.Get("keepdims", 1) == 1;
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::ReduceMean>(axes, keepdims);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
}
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
