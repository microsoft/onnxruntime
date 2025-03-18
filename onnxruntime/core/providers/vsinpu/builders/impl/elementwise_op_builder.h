
/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
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
#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
namespace onnxruntime {
namespace vsi {
namespace npu {
#define ELEMENTWISE_OP_BUILDER(onnx_op_type, vsinpu_op_kind)                                      \
  class onnx_op_type##OpBuilder : public BaseOpBuilder {                                          \
    bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              \
                       const Node* node) const override {                                         \
      for (auto input : node->InputDefs()) {                                                      \
        if (*input->Type() == "tensor(int64)") {                                                  \
          LOGS_DEFAULT(WARNING) << "Int64 type is not supported as elementwise operation input."; \
          return false;                                                                           \
        }                                                                                         \
      }                                                                                           \
      return true;                                                                                \
    }                                                                                             \
    bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,                                               \
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,                     \
                       std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,                    \
                       const NodeUnit& node_unit) override {                                      \
      LOGS_DEFAULT(INFO) << "Creating " << #onnx_op_type << " Op";                                \
      auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::vsinpu_op_kind>();            \
      (*op).BindInputs(inputs).BindOutputs(outputs);                                              \
      return true;                                                                                \
      ;                                                                                           \
    }                                                                                             \
  };

ELEMENTWISE_OP_BUILDER(Add, Add);
ELEMENTWISE_OP_BUILDER(Sub, Sub);
ELEMENTWISE_OP_BUILDER(Mul, Multiply);
ELEMENTWISE_OP_BUILDER(Div, Div);  // not consider zero
ELEMENTWISE_OP_BUILDER(Abs, Abs);
ELEMENTWISE_OP_BUILDER(Sqrt, Sqrt);
ELEMENTWISE_OP_BUILDER(Exp, Exp);
ELEMENTWISE_OP_BUILDER(Floor, Floor);
ELEMENTWISE_OP_BUILDER(Log, Log);
ELEMENTWISE_OP_BUILDER(Sin, Sin);
ELEMENTWISE_OP_BUILDER(HardSwish, HardSwish);
ELEMENTWISE_OP_BUILDER(Neg, Neg);
ELEMENTWISE_OP_BUILDER(Not, LogicalNot);
ELEMENTWISE_OP_BUILDER(Ceil, Ceil);
ELEMENTWISE_OP_BUILDER(Round, Round);
ELEMENTWISE_OP_BUILDER(Min, Minimum);
ELEMENTWISE_OP_BUILDER(Max, Maximum);

class PowOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input0_type = *node->InputDefs()[0]->Type();
    auto input1_type = *node->InputDefs()[1]->Type();
    if (input0_type != input1_type) {
      if ((input0_type == "tensor(float)" && input1_type == "tensor(int32)") ||
          (input0_type == "tensor(int32)" && input1_type == "tensor(float)")) {
        LOGS_DEFAULT(WARNING) << "Pow op does not support one of input is float32 while the other one is int32 type.";
        return false;
      }
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(INFO) << "Creating Pow Op";
    auto op = graph_ep->GetGraph()
                  ->CreateOperation<tim::vx::ops::Pow>();
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
