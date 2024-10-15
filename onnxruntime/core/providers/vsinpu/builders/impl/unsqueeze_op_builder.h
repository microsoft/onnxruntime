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
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class UnsqueezeOpBuilder : public BaseOpBuilder {
  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    auto input_type = node_unit.Inputs()[0].node_arg.Type();
    if (*input_type == "tensor(int64)" || !util::IsTypeSupported(&node_unit.Inputs()[0].node_arg)) {
      LOGS_DEFAULT(WARNING) << node_unit.OpType() << " has unsupported input type : "
                            << *input_type;
      return false;
    }
    if (node_unit.SinceVersion() > 11 && !Contains(initializers, node_unit.Inputs()[1].node_arg.Name())) {
      LOGS_DEFAULT(WARNING) << "Only support const axes in Unsqueeze op.";
      return false;
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(INFO) << "Creating Unsqueeze Op.";

    NodeAttrHelper helper(node_unit.GetNode());
    std::vector<int64_t> def_axes;
    auto input_shape_size = inputs[0]->GetShape().size();

    if (node_unit.SinceVersion() < 13 && helper.HasAttr("axes")) {
      def_axes = helper.Get("axes", def_axes);
    } else if (inputs.size() > 1) {
      def_axes.resize(inputs[1]->GetSpec().GetElementNum());
      inputs[1]->CopyDataFromTensor(def_axes.data());
    } else {  // if axes is empty from onnx, check input shape to determine
      for (int64_t i = 0; i < input_shape_size; ++i) {
        if (inputs[0]->GetShape()[i] == 1) {
          def_axes.push_back(i);
        }
      }
    }

    std::vector<int32_t> axes(def_axes.begin(), def_axes.end());
    axes = util::ReverseAxis(axes, input_shape_size + axes.size());

    std::vector<uint32_t> timvx_axes(inputs[0]->GetShape().begin(), inputs[0]->GetShape().end());
    for (int32_t dim : axes) {
      timvx_axes.insert(timvx_axes.begin() + dim, 1);
    }

    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(timvx_axes);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
