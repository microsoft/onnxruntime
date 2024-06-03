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
#include "core/providers/shared/utils/utils.h"
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
namespace onnxruntime {
namespace vsi {
namespace npu {
class GemmOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    NodeAttrHelper helper(*node);
    auto weight_units = helper.Get("transB", 0) == 1 ? vsi::npu::util::GetTensorShape(*input_defs[1]).GetDims()[0] : vsi::npu::util::GetTensorShape(*input_defs[1]).GetDims()[1];
    if (input_defs.size() > 2) {
      auto bias_shape = vsi::npu::util::GetTensorShape(*input_defs[2]);
      if (bias_shape.NumDimensions() == 1 && bias_shape.GetDims()[0] != weight_units) {
        LOGS_DEFAULT(WARNING) << "Not support to broadcast bias shape.";
        return false;
      } else if (bias_shape.NumDimensions() == 2 && (bias_shape.Size() != weight_units || (bias_shape.GetDims()[0] != 1 && bias_shape.GetDims()[1] != 1))) {
        LOGS_DEFAULT(WARNING) << "Not support 2-dims bias shape.";
        return false;
      }

      if (*input_defs[2]->Type() == "tensor(float16)" && !graph_viewer.IsConstantInitializer(input_defs[2]->Name(), true)) {
        LOGS_DEFAULT(WARNING) << "Not support f16 bias with input attr.";
        return false;
      }
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Gemm Op.";
    auto input_A = inputs[0];
    auto input_B = inputs[1];
    NodeAttrHelper helper(node_unit.GetNode());

    auto trans_A = helper.Get("transA", 0);
    auto trans_B = helper.Get("transB", 0);
    const bool has_alpha = (helper.Get("alpha", 1.0f) != 1.0);
    const bool has_beta = (helper.Get("beta", 1.0f) != 1.0);
    const bool has_C = (inputs.size() == 3);
    auto weight_units = helper.Get("transB", 0) == 1 ? inputs[1]->GetShape()[1] : inputs[1]->GetShape()[0];

    tim::vx::TensorSpec coef_spec(tim::vx::DataType::FLOAT32, {1},
                                  tim::vx::TensorAttribute::CONSTANT);

    auto multiply_impl = [&](std::shared_ptr<tim::vx::Tensor> input,
                             std::shared_ptr<tim::vx::Tensor> coef,
                             std::shared_ptr<tim::vx::Tensor> output) {
      auto multiply_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Multiply>();
      (*multiply_op).BindInput(input).BindInput(coef).BindOutput(output);
      graph_ep->GetOps().push_back(multiply_op);
    };

    auto transpose_impl = [&](std::shared_ptr<tim::vx::Tensor> input,
                              std::shared_ptr<tim::vx::Tensor> output) {
      std::vector<uint32_t> perm = {1U, 0U};
      auto transpose_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Transpose>(perm);
      (*transpose_op).BindInput(input).BindOutput(output);
      graph_ep->GetOps().push_back(std::move(transpose_op));
    };

    auto fc_impl = [&](std::vector<std::shared_ptr<tim::vx::Tensor>> inputs,
                       std::shared_ptr<tim::vx::Tensor> output) {
      auto fc_op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::FullyConnected>(0, weight_units);
      (*fc_op).BindInputs(inputs).BindOutput(output);
      graph_ep->GetOps().push_back(std::move(fc_op));
    };

    auto alpha_A = input_A;
    std::shared_ptr<tim::vx::Tensor> beta_C;
    auto final_A = input_A;
    auto final_B = input_B;

    if (has_alpha) {
      auto alpha_tensor = graph_ep->GetGraph()->CreateTensor(coef_spec);
      auto alpha = helper.Get("alpha", 1.0f);
      alpha_tensor->CopyDataToTensor(&alpha);
      alpha_A = graph_ep->GetGraph()->CreateTensor(
          input_A->GetSpec().AsTransientSpec());
      multiply_impl(input_A, alpha_tensor, alpha_A);
      final_A = alpha_A;
    }
    if (has_beta) {
      auto beta_tensor = graph_ep->GetGraph()->CreateTensor(coef_spec);
      auto beta = helper.Get("beta", 1.0f);
      beta_tensor->CopyDataToTensor(&beta);
      beta_C = graph_ep->GetGraph()->CreateTensor(
          inputs[2]->GetSpec().AsTransientSpec());
      multiply_impl(inputs[2], beta_tensor, beta_C);
    } else if (has_C) {
      beta_C = inputs[2];
    }

    if (trans_A) {
      final_A = graph_ep->GetGraph()->CreateTensor(
          input_A->GetSpec().AsTransientSpec());
      transpose_impl(alpha_A, final_A);
    }
    if (!trans_B) {
      final_B = graph_ep->GetGraph()->CreateTensor(
          input_B->GetSpec().AsTransientSpec());
      transpose_impl(input_B, final_B);
    }
    std::vector<std::shared_ptr<tim::vx::Tensor>> fc_inputs = {final_A, final_B};

    if (has_C) fc_inputs.push_back(beta_C);
    fc_impl(fc_inputs, outputs[0]);

    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
