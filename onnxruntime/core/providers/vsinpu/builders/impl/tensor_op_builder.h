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
class ReshapeOpBuilder : public BaseOpBuilder {
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 5; }

  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    auto input = node_unit.Inputs()[0];
    auto shape = node_unit.Inputs()[1];
    if (initializers.end() == initializers.find(shape.node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Target shape of reshape op must be known.";
      return false;
    }
    if (util::IsTypeSupported(&input.node_arg) && util::IsTypeSupported(&shape.node_arg)) {
      if (*input.node_arg.Type() != "tensor(int64)") {
        return true;
      }
    }
    return false;
  }

  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();

    NodeAttrHelper helper(*node);
    const bool allow_zero = helper.Get("allowzero", 0) == 1;
    auto& perm_tensor_proto = *graph_viewer.GetConstantInitializer(input_defs[1]->Name(), true);
    std::vector<int64_t> perm(perm_tensor_proto.dims()[0]);
    auto status = onnxruntime::utils::UnpackTensor(
        perm_tensor_proto,
        perm_tensor_proto.has_raw_data() ? perm_tensor_proto.raw_data().data() : nullptr,
        perm_tensor_proto.has_raw_data() ? perm_tensor_proto.raw_data().size() : 0,
        perm.data(), perm.size());

    // Check if perm has any 0's when allow zero is enabled.
    if (allow_zero && std::find(perm.begin(), perm.end(), 0L) != perm.end()) {
      LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 as dimension when allowzero is enabled";
      return false;
    }

    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Reshape Op.";
    std::vector<int64_t> new_shape(inputs[1]->GetShape()[0]);
    inputs[1]->CopyDataFromTensor(new_shape.data());
    for (size_t i = 0; i < new_shape.size(); i++) {
      if (new_shape[i] == 0) {
        new_shape[i] = inputs[0]->GetShape()[inputs[0]->GetShape().size() - i - 1];
      }
    }

    int64_t element_count = std::accumulate(new_shape.begin(), new_shape.end(), static_cast<int64_t>(1),
                                            [&](int64_t a, int64_t b) {
                                              return b == -1 ? a : a * b;
                                            });
    auto negative_it = std::find(new_shape.begin(), new_shape.end(), -1);
    if (negative_it != new_shape.end()) {
      *negative_it = inputs[0]->GetSpec().GetElementNum() / element_count;
    }

    std::vector<uint32_t> new_shape_uint32(new_shape.begin(), new_shape.end());
    std::reverse(new_shape_uint32.begin(), new_shape_uint32.end());
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Reshape>(new_shape_uint32);
    (*op).BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

class TransposeOpBuilder : public BaseOpBuilder {
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    auto shape_dim = vsi::npu::util::GetTensorShape(*input_defs[0]).NumDimensions();
    NodeAttrHelper helper(*node);
    auto perm = helper.Get("perm", std::vector<uint32_t>(shape_dim, 1));
    if (perm.size() != shape_dim) {
      LOGS_DEFAULT(VERBOSE) << "Size mismatch between perm vector and input shape.";
      return false;
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Transpose Op.";
    std::vector<int64_t> def_val(inputs[0]->GetShape().size());
    for (int64_t i = 0; i < def_val.size(); i++) def_val[i] = def_val.size() - i - 1;

    NodeAttrHelper helper(node_unit.GetNode());
    def_val = helper.Get("perm", def_val);
    std::vector<uint32_t> timvx_perm;
    for (uint32_t i = 0; i < def_val.size(); i++) {
      timvx_perm.push_back(def_val.size() - 1 - def_val[def_val.size() - i - 1]);
    }
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Transpose>(timvx_perm);
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
