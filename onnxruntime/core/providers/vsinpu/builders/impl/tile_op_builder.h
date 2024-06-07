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

namespace onnxruntime {
namespace vsi {
namespace npu {
class TileOpBuilder : public BaseOpBuilder {
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }

  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    auto input = node_unit.Inputs()[0];
    auto multipliers = node_unit.Inputs()[1];
    if (initializers.end() == initializers.find(multipliers.node_arg.Name())) {
      LOGS_DEFAULT(WARNING) << "Multipliers of tile op must be known.";
      return false;
    }
    if (util::IsTypeSupported(&input.node_arg) && util::IsTypeSupported(&multipliers.node_arg)) {
      if (*input.node_arg.Type() != "tensor(int64)") {
        return true;
      }
    }
    LOGS_DEFAULT(WARNING) << "Input type not supported.";
    return false;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Tile Op.";
    std::vector<int64_t> multipliers(inputs[1]->GetShape()[0]);
    inputs[1]->CopyDataFromTensor(multipliers.data());
    std::reverse(multipliers.begin(), multipliers.end());
    std::vector<int32_t> timvx_multipliers(multipliers.begin(), multipliers.end());
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Tile>(timvx_multipliers);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
