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
#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class GatherOpBuilder : public BaseOpBuilder {
  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    auto input = node_unit.Inputs()[0];
    auto indices = node_unit.Inputs()[1];
    if (util::IsTypeSupported(&input.node_arg) && util::IsTypeSupported(&indices.node_arg)) {
      if (*input.node_arg.Type() == "tensor(int64)") {
        LOGS_DEFAULT(WARNING) << "Only support indices tensor to be int64 type in gather op.";
        return false;
      }
      if (*indices.node_arg.Type() != "tensor(int64)" && *indices.node_arg.Type() != "tensor(int32)") {
        LOGS_DEFAULT(WARNING) << "Unsupported indices tensor type in gather op.";
        return false;
      }
      if (*indices.node_arg.Type() == "tensor(int64)" && !Contains(initializers, indices.node_arg.Name())) {
        LOGS_DEFAULT(WARNING) << "Only support const attribute if indice tensor is in int64 type.";
        return false;
      }
      return true;
    }
    return false;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Gather Op.";
    auto indices = node_unit.Inputs()[1];
    int8_t is_scalar_indices = 0;
    NodeAttrHelper helper(node_unit.GetNode());
    auto axis = helper.Get("axis", 0);
    axis = util::ReverseAxis(axis, inputs[0]->GetShape().size());
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Gather>(axis, 0);
    auto indices_shape_proto = indices.node_arg.Shape();
    if (indices_shape_proto != nullptr) {
      if (indices_shape_proto->dim_size() == 0) {
        is_scalar_indices = 1;
      }
    } else {
      is_scalar_indices = 1;
    }

    bool is_i64_indices = inputs[1]->GetDataType() == tim::vx::DataType::INT64;
    if (!is_i64_indices) {
      inputs[1]->SetScalar(is_scalar_indices);
      (*op).BindInputs(inputs).BindOutputs(outputs);
    } else {
      std::vector<int64_t> origin_data(inputs[1]->GetSpec().GetElementNum());
      inputs[1]->CopyDataFromTensor(origin_data.data());
      std::vector<int32_t> transformed_data(origin_data.begin(), origin_data.end());
      tim::vx::TensorSpec ts = inputs[1]->GetSpec();
      ts.SetDataType(tim::vx::DataType::INT32);
      auto transformed_indices = graph_ep->GetGraph()->CreateTensor(ts, transformed_data.data());
      transformed_indices->SetScalar(is_scalar_indices);
      (*op).BindInput(inputs[0]).BindInput(transformed_indices).BindOutput(outputs[0]);
    }
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
