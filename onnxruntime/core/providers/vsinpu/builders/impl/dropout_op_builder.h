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
#ifndef ONNXRUNTIME_CORE_PROVIDERS_VSINPU_BUILDERS_IMPL_DROPOUT_OP_BUILDER_H_
#define ONNXRUNTIME_CORE_PROVIDERS_VSINPU_BUILDERS_IMPL_DROPOUT_OP_BUILDER_H_
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class DropoutOpBuilder : public BaseOpBuilder {
  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    if (node_unit.Inputs().size() > 2) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto =
          initializers.at(node_unit.Inputs()[2].node_arg.Name());
      std::vector<uint8_t> training_mode(1);
      auto status = onnxruntime::utils::UnpackTensor(
          *tensor_proto,
          tensor_proto->has_raw_data() ? tensor_proto->raw_data().data() : nullptr,
          tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0,
          training_mode.data(), training_mode.size());
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Failed to get data training mode tensor.";
        return false;
      }
      if (training_mode[0] == true) {
        LOGS_DEFAULT(WARNING) << "Only support inference typed dropout now.";
        return false;
      }
    }
    if (node_unit.Inputs().size() > 1) return false;
    return true;
  }
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    if (helper.HasAttr("seed")) {
      LOGS_DEFAULT(WARNING) << "Not support seed in Dropout op.";
      return false;
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating DropOut Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Dropout>(1.0);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
#endif  // ONNXRUNTIME_CORE_PROVIDERS_VSINPU_BUILDERS_IMPL_DROPOUT_OP_BUILDER_H_
