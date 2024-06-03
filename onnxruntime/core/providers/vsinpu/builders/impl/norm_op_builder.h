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

class BatchNormOpBuilder : public BaseOpBuilder {
  enum NormINPUTS {
    input_tensor = 0,
    scale_tensor = 1,
    Bias_tensor = 2,
    mean_tensor = 3,
    var_tensor = 4
  };
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override{ return 9; }

  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    NodeAttrHelper helper(*node);
    auto training_mode = helper.Get("training_mode", 0);
    if (training_mode) {
      LOGS_DEFAULT(WARNING) << "Training is not supported in batch_norm op.";
      return false;
    }
    if (helper.HasAttr("spatial")) {
      LOGS_DEFAULT(WARNING) << "VSINPU does not support 'spatial' parameter.";
      return false;
    }
    if (!graph_viewer.IsConstantInitializer(input_defs[NormINPUTS::scale_tensor]->Name(), true)) {
      LOGS_DEFAULT(WARNING) << "Not support mean/var/gamma/beta set as dynamic input yet.";
      return false;
    }

    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(INFO) << "Creating BatchNorm Op.";
    NodeAttrHelper helper(node_unit.GetNode());
    auto epsilon = helper.Get("epsilon", 1e-5f);
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::BatchNorm>(epsilon);
    std::vector<std::shared_ptr<tim::vx::Tensor>> reordered_inputs;
    int indices[] = {NormINPUTS::input_tensor, NormINPUTS::mean_tensor, NormINPUTS::var_tensor, NormINPUTS::scale_tensor, NormINPUTS::Bias_tensor};
    for (int i : indices) {
      reordered_inputs.push_back(inputs[i]);
    }
    (*op).BindInputs(reordered_inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
