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
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace vsi {
namespace npu {
class ResizeOpBuilder : public BaseOpBuilder {
  bool HasSupportedInputOutputsImpl(const InitializedTensorSet& initializers,
                                    const NodeUnit& node_unit) const override {
    auto input_type = node_unit.Inputs()[0].node_arg.Type();
    if (*input_type == "tensor(int64)" || !util::IsTypeSupported(&node_unit.Inputs()[0].node_arg)) {
      LOGS_DEFAULT(WARNING) << node_unit.OpType() << " has unsupported input type : "
                            << *input_type;
      return false;
    }
    if (node_unit.SinceVersion() > 10) {
      if (node_unit.Inputs().size() > 2 && !Contains(initializers, node_unit.Inputs()[2].node_arg.Name())) {
        LOGS_DEFAULT(WARNING) << "Scale tensor must be constant.";
        return false;
      }
      if (node_unit.Inputs().size() > 3 && !Contains(initializers, node_unit.Inputs()[3].node_arg.Name())) {
        LOGS_DEFAULT(WARNING) << "Size tensor must be constant.";
        return false;
      }
    } else {
      if (!Contains(initializers, node_unit.Inputs()[1].node_arg.Name())) {
        LOGS_DEFAULT(WARNING) << "Scale tensor must be constant.";
        return false;
      }
    }
    return true;
  }
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer, const Node* node) const override {
    auto shape = vsi::npu::util::GetTensorShape(*node->InputDefs()[0]);
    if (shape.NumDimensions() > 4) {
      LOGS_DEFAULT(WARNING) << "3D or more dimesions resize is not supported.";
      return false;
    }

    NodeAttrHelper helper(*node);
    if (helper.Get("antialiax", 0) != 0) {
      LOGS_DEFAULT(WARNING) << "Antialias attribute is not supported.";
      return false;
    }
    auto& cooridinate = helper.Get("coordinate_transoformation_mode", "half_pixel");
    if (cooridinate != "align_corners" && cooridinate != "half_pixel" && cooridinate != "half_pixel_symmetric") {
      LOGS_DEFAULT(WARNING) << "Only support half_pixel_symmetric and align_corners attributes now.";
      return false;
    }
    if (helper.Get("keep_aspect_ratio_policy", "stretch") != "stretch") {
      LOGS_DEFAULT(WARNING) << "Not support to keep aspect ratio.";
      return false;
    }
    if (helper.Get("mode", "nearest") == "cubic") {
      LOGS_DEFAULT(WARNING) << "Not support the cubic resize type yet.";
      return false;
    }
    if (helper.HasAttr("axes")) {
      LOGS_DEFAULT(WARNING) << "Axes-specifying is not support.";
      return false;
    }
    return true;
  }

  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating Resize Op.";
    auto inputs_num = inputs.size();
    bool is_1dresize = inputs[0]->GetShape().size() == 1;
    NodeAttrHelper helper(node_unit.GetNode());
    auto onnx_mode = helper.Get("mode", "nearest");
    auto coordinate_transformation = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool is_size_set = helper.HasAttr("size");
    int32_t scale_index = node_unit.SinceVersion() > 10 ? 2 : 1;

    auto resize_type = onnx_mode == "nearest" ? tim::vx::ResizeType::NEAREST_NEIGHBOR : tim::vx::ResizeType::BILINEAR;
    bool align_corners = coordinate_transformation == "align_corners";
    bool half_pixel_center = coordinate_transformation == "half_pixel_symmetric";
    std::shared_ptr<tim::vx::Operation> op = nullptr;
    if (is_1dresize) {
      int target_size;
      if (is_size_set) {
        int64_t onnx_size;
        inputs[3]->CopyDataFromTensor(&onnx_size);
        target_size = static_cast<int>(onnx_size);
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Resize1d>(resize_type, 0.0f, align_corners, half_pixel_center, target_size);
      } else {
        float scale;
        inputs[scale_index]->CopyDataFromTensor(&scale);
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Resize1d>(resize_type, scale, align_corners, half_pixel_center, 0);
      }
    } else {
      int target_height, target_width;
      if (is_size_set) {
        std::vector<int64_t> onnx_sizes(inputs[3]->GetShape().size());
        inputs[3]->CopyDataFromTensor(onnx_sizes.data());
        target_height = static_cast<int>(onnx_sizes[1]);
        target_width = static_cast<int>(onnx_sizes[0]);
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Resize>(resize_type, 0.0f, align_corners, half_pixel_center, target_height, target_width);
      } else {
        auto input_shape = inputs[0]->GetShape();
        std::vector<float> scales(input_shape.size());
        std::vector<uint32_t> out_shape(input_shape.size());
        inputs[scale_index]->CopyDataFromTensor(scales.data());
        for (int i = 0; i < input_shape.size(); i++) {
          out_shape[i] = input_shape[i] * scales[input_shape.size() - 1 - i];
        }
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Resize>(resize_type, 0, align_corners, half_pixel_center, out_shape[1], out_shape[0]);
      }
    }

    (*op).BindInput(inputs[0]).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
