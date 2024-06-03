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

namespace onnxruntime {
namespace vsi {
namespace npu {
class BaseQLinearOpBuilder : public BaseOpBuilder {
  enum {
    INPUT_A = 0,
    INPUT_A_SCALE = 1,
    INPUT_A_ZP = 2,
    INPUT_B = 3,
    INPUT_B_SCALE = 4,
    INPUT_B_ZP = 5,
    OUTPUT_SCALE = 6,
    OUTPUT_ZP = 7,
  };

 protected:
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, const Node* node) const override {
    for (int i = 0; i < node->InputDefs().size(); i++) {
      if (i == INPUT_A || i == INPUT_B) continue;
      if (!graph_viewer.IsConstantInitializer(node->InputDefs()[i]->Name(), true)) {
        LOGS_DEFAULT(WARNING) << "Only support const scale / zero point.";
        return false;
      }
    }
    return true;
  }
};

class QLinearAddOpBuilder : public BaseQLinearOpBuilder {
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearAdd Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Add>();
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

class QLinearMulOpBuilder : public BaseQLinearOpBuilder {
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearMul Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Multiply>();
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
