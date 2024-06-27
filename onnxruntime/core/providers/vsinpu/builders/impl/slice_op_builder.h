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
#ifndef ONNXRUNTIME_CORE_PROVIDERS_VSINPU_BUILDERS_IMPL_SLICE_OP_BUILDER_H_
#define ONNXRUNTIME_CORE_PROVIDERS_VSINPU_BUILDERS_IMPL_SLICE_OP_BUILDER_H_
#include <memory>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
enum SliceInputs {
  data = 0,
  starts = 1,
  ends = 2,
  axes = 3,
  steps = 4
};

class SliceOpBuilder : public BaseOpBuilder {
 public:
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 10; }

  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    for (size_t i = 0; i < node_unit.Inputs().size(); ++i) {
      const auto& iodef = node_unit.Inputs()[i];
      if (!util::IsTypeSupported(&iodef.node_arg) ||
          (i == 0 && *iodef.node_arg.Type() == "tensor(int64)") ||
          (i != 0 && !Contains(initializers, iodef.node_arg.Name()))) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  void CopyTensorDataToVector(const std::shared_ptr<tim::vx::Tensor>& tensor, std::vector<int32_t>& vec) {
    std::vector<T> data(tensor->GetSpec().GetElementNum());
    tensor->CopyDataFromTensor(data.data());
    std::transform(data.begin(), data.end(), vec.begin(), [](T val) {
      return static_cast<int32_t>(std::clamp(val, static_cast<T>(std::numeric_limits<int32_t>::min()),
                                             static_cast<T>(std::numeric_limits<int32_t>::max())));
    });
  }

  void ProcessAxes(const std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                   int dims, bool full_axes,
                   std::vector<int32_t>& timvx_starts,
                   std::vector<int32_t>& timvx_ends,
                   std::vector<int32_t>& timvx_strides) {
    auto num_elements = full_axes ? dims : inputs[SliceInputs::axes]->GetSpec().GetElementNum();
    std::vector<int32_t> onnx_starts(num_elements), onnx_ends(num_elements),
        onnx_axes(num_elements), onnx_strides(num_elements, 1);

    auto data_type = inputs[SliceInputs::starts]->GetSpec().GetDataType();
    std::iota(onnx_axes.begin(), onnx_axes.end(), 0);
    if (data_type == tim::vx::DataType::INT64) {
      CopyTensorDataToVector<int64_t>(inputs[SliceInputs::starts], onnx_starts);
      CopyTensorDataToVector<int64_t>(inputs[SliceInputs::ends], onnx_ends);
      if (inputs.size() > 3) {
        CopyTensorDataToVector<int64_t>(inputs[SliceInputs::axes], onnx_axes);
        if (inputs.size() == 5) {
          CopyTensorDataToVector<int64_t>(inputs[SliceInputs::steps], onnx_strides);
        }
      }
    } else {
      CopyTensorDataToVector<int32_t>(inputs[SliceInputs::starts], onnx_starts);
      CopyTensorDataToVector<int32_t>(inputs[SliceInputs::ends], onnx_ends);
      if (inputs.size() > 3) {
        CopyTensorDataToVector<int32_t>(inputs[SliceInputs::axes], onnx_axes);
        if (inputs.size() == 5) {
          CopyTensorDataToVector<int32_t>(inputs[SliceInputs::steps], onnx_strides);
        }
      }
    }

    if (!full_axes) {
      for (auto& axis : onnx_axes) {
        axis = HandleNegativeAxis(axis, inputs[0]->GetShape().size());
      }
    }

    for (int i = 0; i < dims; ++i) {
      if (full_axes || std::find(onnx_axes.begin(), onnx_axes.end(), i) != onnx_axes.end()) {
        int axes_index = std::distance(onnx_axes.begin(), std::find(onnx_axes.begin(), onnx_axes.end(), i));
        timvx_starts[i] = onnx_starts[axes_index];
        timvx_ends[i] = onnx_ends[axes_index];
        if (inputs.size() == 5) {
          timvx_strides[i] = onnx_strides[axes_index];
        }
      } else if (!full_axes) {
        timvx_starts[i] = 0;
        timvx_ends[i] = inputs[SliceInputs::data]->GetShape()[dims - i - 1];
      }
    }
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Slice Op.";
    auto total_dims = inputs[SliceInputs::data]->GetShape().size();
    bool full_axes = inputs.size() <= 3 || (inputs[SliceInputs::axes]->GetSpec().GetElementNum() == total_dims);
    std::vector<int32_t> timvx_starts(total_dims), timvx_ends(total_dims), timvx_strides(total_dims, 1);

    ProcessAxes(inputs, total_dims, full_axes, timvx_starts, timvx_ends, timvx_strides);

    std::reverse(timvx_starts.begin(), timvx_starts.end());
    std::reverse(timvx_ends.begin(), timvx_ends.end());
    std::reverse(timvx_strides.begin(), timvx_strides.end());

    auto op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::StridedSlice>(
        timvx_starts, timvx_ends, timvx_strides, 0, 0, 0);
    op->BindInput(inputs[SliceInputs::data]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu
}  // namespace vsi
}  // namespace onnxruntime
#endif  // ONNXRUNTIME_CORE_PROVIDERS_VSINPU_BUILDERS_IMPL_SLICE_OP_BUILDER_H_
