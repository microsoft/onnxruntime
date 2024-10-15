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
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class FlattenOpBuilder : public BaseOpBuilder {
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Flatten Op.";
    std::vector<uint32_t> reshape_param;
    if (outputs[0]->GetShape().size() == 2) {
      reshape_param = outputs[0]->GetShape();
    } else {
      auto input_shape = inputs[0]->GetShape();
      NodeAttrHelper helper(node_unit.GetNode());
      int64_t axis = helper.Get("axis", 1);
      axis = util::ReverseAxis(static_cast<int32_t>(axis), input_shape.size());
      uint32_t first_dim = 1;
      for (int64_t i = 0; i < axis; i++) {
        first_dim *= inputs[0]->GetShape()[i];
      }
      uint32_t second_dim = inputs[0]->GetSpec().GetElementNum() / first_dim;
      reshape_param.push_back(first_dim);
      reshape_param.push_back(second_dim);
    }
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(reshape_param);
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
