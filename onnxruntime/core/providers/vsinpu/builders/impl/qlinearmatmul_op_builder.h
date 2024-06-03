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

class QLinearMatMulOpBuilder : public BaseOpBuilder {
  enum {
    matrixA = 0,
    A_scale = 1,
    A_zero_point = 2,
    matrixB = 3,
    B_scale = 4,
    B_zero_point = 5,
    out_scale = 6,
    out_zero_point = 7
  };
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    auto A_def = input_defs[matrixA];
    auto B_def = input_defs[matrixB];
    for (auto def : input_defs) {
      if (def->Name() == A_def->Name() || def->Name() == B_def->Name())
        continue;
      else {
        if (!graph_viewer.IsConstantInitializer(def->Name(), true)) {
          LOGS_DEFAULT(WARNING) << "Scale and zero point must be known before setting graph.";
          return false;
        }
      }
    }
    int64_t A_elements = util::GetTensorShape(*input_defs[A_scale]).Size();
    int64_t B_elements = util::GetTensorShape(*input_defs[B_scale]).Size();
    int64_t Out_elements = util::GetTensorShape(*input_defs[out_scale]).Size();
    if (A_elements > 1 || B_elements > 1 || Out_elements > 1) {
      LOGS_DEFAULT(WARNING) << "Per channel quantized input/output is not supported in QLinearMatmul Op.";
      return false;
    }

    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(INFO) << "Creating QLinearMatmul Op.";
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Matmul>();
    (*op).BindInputs(inputs).BindOutputs(outputs);
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
