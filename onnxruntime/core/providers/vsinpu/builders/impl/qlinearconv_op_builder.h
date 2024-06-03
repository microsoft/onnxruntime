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
#include "core/providers/shared/utils/utils.h"
#include "core/providers/vsinpu/builders/impl/base_op_builder.h"
#include "core/framework/tensorprotoutils.h"
#include <variant>
namespace onnxruntime {
namespace vsi {
namespace npu {
class QLinearConvOpBuilder : public BaseOpBuilder {
  enum QLinearConvINPUTS {
    INPUT_TENSOR = 0,
    INPUT_TENSOR_SCALE = 1,
    INPUT_TENSOR_ZP = 2,
    WEIGHT_TENSOR = 3,
    WEIGHT_TENSOR_SCALE = 4,
    WEIGHT_TENSOR_ZP = 5,
    OUTPUT_TENSOR_SCALE = 6,
    OUTPUT_TENSOR_ZP = 7,
    BIAS_TENSOR = 8,
  };

  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    auto input_defs = node->InputDefs();
    auto input_shape = vsi::npu::util::GetTensorShape(*input_defs[QLinearConvINPUTS::INPUT_TENSOR]);
    auto w_scale_shape = vsi::npu::util::GetTensorShape(*input_defs[QLinearConvINPUTS::WEIGHT_TENSOR_SCALE]);
    auto w_shape_dims = vsi::npu::util::GetTensorShape(*input_defs[QLinearConvINPUTS::WEIGHT_TENSOR]).GetDims();
    if (input_shape.NumDimensions() != 4) {
      LOGS_DEFAULT(WARNING) << "Not support conv3d&& conv1d yet.";
      return false;
    }

    if (!graph_viewer.IsConstantInitializer(input_defs[QLinearConvINPUTS::INPUT_TENSOR_SCALE]->Name(), true) || !graph_viewer.IsConstantInitializer(input_defs[WEIGHT_TENSOR]->Name(), true)) {
      LOGS_DEFAULT(WARNING) << "Not support quantization definitions or weights that are not constant yet.";
      return false;
    }

    if (w_shape_dims[2] > 15) {
      LOGS_DEFAULT(WARNING) << "Not support weight kernel with height higher than 15.";
      return false;
    }

    if (w_scale_shape.Size() != 1 && *input_defs[WEIGHT_TENSOR]->Type() == "tensor(int8)") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto =
          graph_viewer.GetConstantInitializer(input_defs[QLinearConvINPUTS::WEIGHT_TENSOR_ZP]->Name(), true);
      std::vector<int8_t> w_zp(tensor_proto->dims_size() == 0 ? 1 : tensor_proto->dims()[0]);

      auto status = onnxruntime::utils::UnpackTensor(
          *tensor_proto,
          tensor_proto->has_raw_data() ? tensor_proto->raw_data().data() : nullptr,
          tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0,
          w_zp.data(), w_zp.size());
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Failed to get data from weight zp tensor.";
        return false;
      }
      if (std::any_of(w_zp.begin(), w_zp.end(), [](int i) { return i != 0; })) {
        LOGS_DEFAULT(WARNING) << "Asymmetric perchannel quantization only allows uint8 datatype or int8 with all zero.";
        return false;
      }
    }
    return true;
  }
  bool HandleBuildOp(vsi::npu::GraphEP* graph_ep,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& inputs,
                     std::vector<std::shared_ptr<tim::vx::Tensor>>& outputs,
                     const NodeUnit& node_unit) override {
    LOGS_DEFAULT(VERBOSE) << "Creating QLinearConv Op.";

    NodeAttrHelper helper(node_unit.GetNode());
    auto padtype = helper.Get("auto_pad", std::string(""));
    auto group = helper.Get("group", static_cast<uint32_t>(1));
    std::vector<uint32_t> default_vec = {1, 1, 1, 1};
    auto stride =
        helper.Get("strides", default_vec);
    auto dilation =
        helper.Get("dilations", default_vec);
    std::shared_ptr<tim::vx::Operation> op;
    if (padtype != "NOTSET") {  // array "pads" is not set
      if (group != 1 && group != inputs[1]->GetShape()[3]) {
        op = graph_ep->GetGraph()
                 ->CreateOperation<tim::vx::ops::GroupedConv2d>(
                     vsi::npu::util::GetPadType(padtype),
                     std::array<uint32_t, 2>{stride[1], stride[0]},
                     std::array<uint32_t, 2>{dilation[1], dilation[0]}, group,
                     tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);

      } else {
        int32_t multiplier = group == 1 ? 0 : inputs[1]->GetShape()[3] / inputs[0]->GetShape()[2];
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
            vsi::npu::util::GetPadType(padtype),
            std::array<uint32_t, 2>{stride[1], stride[0]},
            std::array<uint32_t, 2>{dilation[1], dilation[0]}, multiplier,
            tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
      }
    } else {
      std::vector<uint32_t> default_pads(4, 0);
      auto pads = helper.Get("pads", default_pads);
      if (group != 1 && group != inputs[1]->GetShape()[3]) {
        op = graph_ep->GetGraph()
                 ->CreateOperation<tim::vx::ops::GroupedConv2d>(
                     std::array<uint32_t, 4>{pads[1], pads[3], pads[0], pads[2]},
                     std::array<uint32_t, 2>{stride[1], stride[0]},
                     std::array<uint32_t, 2>{dilation[1], dilation[0]}, group,
                     tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);

      } else {
        int32_t multiplier = group == 1 ? 0 : inputs[1]->GetShape()[3] / inputs[0]->GetShape()[2];
        op = graph_ep->GetGraph()->CreateOperation<tim::vx::ops::Conv2d>(
            std::array<uint32_t, 4>{pads[1], pads[3],
                                    pads[0], pads[2]},
            std::array<uint32_t, 2>{stride[1], stride[0]},
            std::array<uint32_t, 2>{dilation[1], dilation[0]}, multiplier,
            tim::vx::DataLayout::WHCN, tim::vx::DataLayout::WHIcOc);
      }
    }
    (*op).BindInputs(inputs).BindOutputs(outputs);
    graph_ep->GetOps().push_back(std::move(op));
    return true;
  }
};
}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
