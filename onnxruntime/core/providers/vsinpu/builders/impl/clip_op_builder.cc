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
#include <limits>
#include <utility>
#include "core/providers/vsinpu/builders/impl/clip_op_builder.h"

namespace onnxruntime {
namespace vsi {
namespace npu {

namespace clip_internal {
template <typename T>
struct LowMax {
  constexpr static T low() {
    return std::numeric_limits<T>::lowest();
  }
  constexpr static T max() {
    return std::numeric_limits<T>::max();
  }
};
}  // namespace clip_internal

template <typename T>
struct ClipOpBuilder::ClipImpl {
  ClipImpl(vsi::npu::GraphEP* graph_ep, std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
           std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs) {
    T min_default = clip_internal::LowMax<T>::low();
    T max_default = clip_internal::LowMax<T>::max();

    T* min_data = &min_default;
    T* max_data = &max_default;
    std::shared_ptr<tim::vx::Tensor> min_tensor = nullptr;
    std::shared_ptr<tim::vx::Tensor> max_tensor = nullptr;
    if (inputs.size() > 1) {
      min_tensor = inputs[1];
      if (inputs.size() > 2) {
        max_tensor = inputs[2];
      }
    }
    if (min_tensor) {
      min_tensor->CopyDataFromTensor(min_data);
    }
    if (max_tensor) {
      max_tensor->CopyDataFromTensor(max_data);
    }
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Clip>(
        static_cast<float>(*min_data), static_cast<float>(*max_data));
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
  }
};

bool ClipOpBuilder::HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                                  std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                                  std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                                  const NodeUnit& node_unit) {
  LOGS_DEFAULT(INFO) << "Creating Clip Op.";
  if (node_unit.SinceVersion() <= 6) {
    NodeAttrHelper helper(node_unit.GetNode());
    auto min = helper.Get("min", -3.402e+38f);
    auto max = helper.Get("max", 3.402e+38f);
    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Clip>(min, max);
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
  } else {
    switch (inputs[0]->GetDataType()) {
      case tim::vx::DataType::INT8:
        ClipImpl<int8_t>(graph_ep, inputs, outputs);
        break;
      case tim::vx::DataType::UINT8:
        ClipImpl<uint8_t>(graph_ep, inputs, outputs);
        break;
      case tim::vx::DataType::INT16:
        ClipImpl<int16_t>(graph_ep, inputs, outputs);
        break;
      case tim::vx::DataType::INT32:
        ClipImpl<int32_t>(graph_ep, inputs, outputs);
        break;
      case tim::vx::DataType::FLOAT16:
        ClipImpl<Ort::Float16_t>(graph_ep, inputs, outputs);
        break;
      case tim::vx::DataType::FLOAT32:
      default:
        ClipImpl<float>(graph_ep, inputs, outputs);
        break;
    }
  }
  return true;
}

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
