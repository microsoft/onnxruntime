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
class DequantizeLinearOpBuilder : public BaseOpBuilder {
  enum DequantizeINPUTS {
    input_tensor = 0,
    scale_tensor = 1,
    zero_point_tensor = 2
  };
  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {

    auto input_type = node_unit.Inputs()[0].node_arg.Type();
    if (*input_type == "tensor(int64)" || !util::IsTypeSupported(&node_unit.Inputs()[0].node_arg)) {
      LOGS_DEFAULT(WARNING) << node_unit.OpType() << " has unsupported input type : "
                            << *input_type;
      return false;
    }
    if (!node_unit.Inputs()[0].quant_param.has_value()) {
      LOGS_DEFAULT(WARNING) << "The quantization params must be known.";
      return false;
    }
    if (node_unit.Inputs()[0].quant_param->scale.Shape()->dim_size() != 0 && node_unit.Inputs()[0].quant_param->scale.Shape()->dim(0).dim_value() != 1) {
      LOGS_DEFAULT(WARNING) << "Per channel quantized input is not support in DequantizeLinear op.";
      return false;
    }
    return true;
  }
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    if (helper.HasAttr("block_size") && helper.Get("block_size", 0) != 0) {
      LOGS_DEFAULT(WARNING) << "Not support block quantization yet.";
      return false;
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(INFO) << "Creating Dequantize Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::DataConvert>();
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
