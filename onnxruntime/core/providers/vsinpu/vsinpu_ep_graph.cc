
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
#include <algorithm>
#include "core/providers/vsinpu/vsinpu_ep_graph.h"
#include "core/providers/vsinpu/builders/op_builder_factory.h"
#include "core/providers/vsinpu/vsinpu_util.h"
#include "core/framework/node_unit.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"

namespace onnxruntime {

namespace vsi {
namespace npu {
GraphEP::GraphEP(const onnxruntime::GraphViewer& graph_viewer, const logging::Logger& logger)
    : graph_viewer_(graph_viewer), logger_(logger) {
  Prepare();
  context_ = tim::vx::Context::Create();
  graph_ = context_->CreateGraph();
  compiled_ = false;
}

bool GraphEP::Prepare() {
  std::tie(node_unit_holder_, node_unit_map_) = QDQ::GetAllNodeUnits(graph_viewer_, logger_);
  for (const auto& node_unit : node_unit_holder_) {
    auto quant_op_type = util::GetQuantizedOpType(*node_unit);

    // Not a qlinear op or qdq node group
    if (quant_op_type == util::QuantizedOpType::Unknown)
      continue;

    const auto add_quantized_input =
        [&all_quantized_op_inputs = all_quantized_op_inputs_](const NodeUnit& node_unit, size_t input_idx) {
          const auto& input_name = node_unit.Inputs()[input_idx].node_arg.Name();
          all_quantized_op_inputs[input_name].push_back(&node_unit);
        };

    // All quantized ops EXCEPT QuantizeLinear has quantized input
    if (quant_op_type != util::QuantizedOpType::QuantizeLinear) {
      add_quantized_input(*node_unit, 0);
    }

    if (util::IsQuantizedBinaryOp(quant_op_type)) {
      add_quantized_input(*node_unit, 1);
      if (util::IsQuantizedConv(quant_op_type) && node_unit->Inputs().size() == 3) {
        add_quantized_input(*node_unit, 2);
      }
    }
  }  // All quantized inputs is recorded
  return true;
}

bool GraphEP::SupportedOp(const onnxruntime::GraphViewer& graph_viewer,
                          const NodeUnit& node_unit) {
  const auto& supported_builtins = vsi::npu::SupportedBuiltinOps();
  const auto& target_node = node_unit.GetNode();
  const auto& it = supported_builtins.find(target_node.OpType());
  if (supported_builtins.end() != it) {
    return it->second->IsSupported(graph_viewer, node_unit);
  }
  LOGS_DEFAULT(WARNING) << "Fallback unsupported op (node_unit) " << node_unit.OpType()
                        << "  to cpu.";
  return false;
}

bool GraphEP::IsNodeSupportedInGroup(const NodeUnit& node_unit, const GraphViewer& graph_viewer) {
  return SupportedOp(graph_viewer, node_unit);
}

const NodeUnit& GraphEP::GetNodeUnit(const Node* node) const {
  const auto node_unit_it = node_unit_map_.find(node);
  ORT_ENFORCE(node_unit_it != node_unit_map_.end(), "Node does not have corresponding NodeUnit.");
  return *node_unit_it->second;
}

void GraphEP::UpdateTensorMap(const std::string& name, const std::shared_ptr<tim::vx::Tensor>& dst_tensor) {
  auto it = tensors_.find(name);
  if (it != tensors_.end()) {
    it->second = dst_tensor;
  }
  for (auto& IO : graph_inputs_) {
    if (IO->name == name) {
      IO->tensor = dst_tensor;
      break;
    }
  }
  for (auto& IO : graph_outputs_) {
    if (IO->name == name) {
      IO->tensor = dst_tensor;
      break;
    }
  }
}

std::shared_ptr<NodeIOInfo> GraphEP::ConstructNodeIO(const std::shared_ptr<tim::vx::Operation>& op,
                                                     std::vector<NodeArg*> input_arg,
                                                     std::vector<NodeArg*> output_arg) {
  auto info = std::make_shared<vsi::npu::NodeIOInfo>();
  info->op_ = op;
  std::vector<std::string> input_names, output_names;
  if (input_arg.empty()) {
    info->input_names_ = std::vector<std::string>();
  } else {
    input_names.reserve(input_arg.size());
    std::transform(input_arg.begin(), input_arg.end(), std::back_inserter(input_names),
                   [](const NodeArg* node) -> std::string {
                     return node->Name();
                   });
    info->input_names_ = input_names;
  }
  if (output_arg.empty()) {
    info->output_names_ = std::vector<std::string>();
  } else {
    output_names.reserve(output_arg.size());
    std::transform(output_arg.begin(), output_arg.end(), std::back_inserter(output_names),
                   [](const NodeArg* node) -> std::string {
                     return node->Name();
                   });
    info->output_names_ = output_names;
  }

  return info;
}

bool GraphEP::BindTensors(const std::shared_ptr<NodeIOInfo>& nodeio_info) {
  auto op = nodeio_info->op_;
  auto input_names = nodeio_info->input_names_;
  auto output_names = nodeio_info->output_names_;
  if (!input_names.empty()) {
    for (auto& name : input_names) {
      if (tensors_.find(name) == tensors_.end() || tensors_[name] == nullptr) {
        LOGS_DEFAULT(ERROR) << "Input tensor not defined or not found!";
        return false;
      }
      (*op).BindInput(tensors_[name]);
    }
  }
  if (!output_names.empty()) {
    for (auto& name : output_names) {
      if (tensors_.find(name) == tensors_.end() || tensors_[name] == nullptr) {
        LOGS_DEFAULT(ERROR) << "Output tensor not defined or not found!";
        return false;
      }
      (*op).BindOutput(tensors_[name]);
    }
  }
  return true;
}

std::shared_ptr<tim::vx::Tensor> GraphEP::MapTIMVXTensor(
    std::shared_ptr<tim::vx::Graph>& graph, const NodeUnitIODef nudef,
    const NodeUnit& node_unit,
    const GraphViewer* graph_viewer, tim::vx::TensorAttribute attribute) {
  const auto& arg = nudef.node_arg;

  if (tensors_.end() != tensors_.find(nudef.node_arg.Name())) {
    return tensors_.find(arg.Name())->second;
  }
  auto shape = vsi::npu::util::OnnxShapeToTIMVXShape(vsi::npu::util::GetTensorShape(arg));
  std::reverse(shape.begin(), shape.end());
  tim::vx::DataType dt = vsi::npu::util::OnnxDtypeToTIMVXDtype(arg.Type());
  tim::vx::TensorSpec spec = tim::vx::TensorSpec(dt, shape, attribute);

  // Tensors have same name may not have same status of quant_param existence, such as QLinearConv->MaxPool->QLinearConv
  // Maxpool output tensor is not set quantization at first pass
  bool is_qtensor = nudef.quant_param.has_value() || Contains(all_quantized_op_inputs_, arg.Name());
  if (is_qtensor) {
    float scale = 0.0f;
    int32_t zp = 0;
    std::optional<std::vector<float>> scales;
    std::optional<std::vector<int32_t>> zps;
    if (nudef.quant_param.has_value()) {
      util::GetQuantizationScaleAndZeroPoint(graph_viewer_,
                                             nudef, node_unit.ModelPath(),
                                             scale, zp, scales, zps);
    } else {
      auto target_nodeunit = all_quantized_op_inputs_[arg.Name()][0];
      auto qinput = all_quantized_op_inputs_[arg.Name()][0]->Inputs();
      auto it = std::find_if(qinput.begin(), qinput.end(), [&arg](const NodeUnitIODef& nud) {
        return nud.node_arg.Name() == arg.Name();
      });
      bool is_conv_bias = std::distance(qinput.begin(), it) == 2;
      if (!is_conv_bias || it->quant_param.has_value()) {
        util::GetQuantizationScaleAndZeroPoint(graph_viewer_,
                                               *it, target_nodeunit->ModelPath(),
                                               scale, zp, scales, zps);
      } else if (!it->quant_param.has_value()) {
        float in_scale, w_scale;
        int32_t in_zp, w_zp;
        std::optional<std::vector<float>> in_scales, w_scales;
        std::optional<std::vector<int32_t>> in_zps, w_zps;

        // onnx defines conv bias with non quantization, but it must be quantized in VSINPU support
        // The bias scale is set as input_scale * weight_scale if per layer quantized,
        // otherwise input_scale* weight_scale[i] if per channel quantized
        util::GetQuantizationScaleAndZeroPoint(graph_viewer_,
                                               qinput[0], target_nodeunit->ModelPath(),
                                               in_scale, in_zp, in_scales, in_zps);
        util::GetQuantizationScaleAndZeroPoint(graph_viewer_,
                                               qinput[1], target_nodeunit->ModelPath(),
                                               w_scale, w_zp, w_scales, w_zps);
        scale = in_scale * w_scale;
        zp = 0;
        if (w_scales) {
          std::vector<float> temp;
          for (size_t i = 0; i < w_scales->size(); i++) {
            temp.push_back(w_scales.value()[i] * in_scale);
          }
          scales = temp;
        }
      }
    }
    tim::vx::Quantization quant;
    // per tensor quantization
    if (!scales.has_value()) {
      quant.SetType(tim::vx::QuantType::ASYMMETRIC);
      quant.SetScales({scale});
      quant.SetZeroPoints({zp});
    } else {  // per channel quantization
      if (zps.has_value()) {
        bool has_nonzero = std::find_if(zps->begin(), zps->end(), [](int elem) { return elem != 0; }) != zps->end();
        if (has_nonzero && *arg.Type() == "tensor(uint8)") {
          quant.SetType(tim::vx::QuantType::ASYMMETRIC_PER_CHANNEL);
        } else {
          quant.SetType(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL);
        }
        quant.SetZeroPoints(zps.value());
      } else {
        if (*arg.Type() == "tensor(int32)" || zp == 0) {
          // set bias quant type
          quant.SetType(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL);
        } else {
          quant.SetType(tim::vx::QuantType::ASYMMETRIC_PER_CHANNEL);
        }
        quant.SetZeroPoints({zp});
      }
      quant.SetScales(scales.value());
      quant.SetChannelDim(shape.size() - 1);
    }
    spec.SetQuantization(quant);
  }

  std::shared_ptr<tim::vx::Tensor> tensor;
  if (attribute ==
      tim::vx::TensorAttribute::CONSTANT) {  // create const tensor
    const ONNX_NAMESPACE::TensorProto* tensor_proto =
        graph_viewer_.GetConstantInitializer(arg.Name(), true);
    std::shared_ptr<uint8_t> unpackedTensor =
        vsi::npu::util::UnpackTensor(&arg, *tensor_proto);

    const void* valueAddr =
        reinterpret_cast<const void*>(unpackedTensor.get());
    tensor = graph->CreateTensor(spec, valueAddr);

  } else {
    tensor = graph->CreateTensor(spec);
  }
  for (auto& input : graph_inputs_) {
    if (input->name == arg.Name()) {
      input->tensor = tensor;
      input->shape = vsi::npu::util::GetTensorShape(arg);
      break;
    }
  }
  for (auto& output : graph_outputs_) {
    if (output->name == arg.Name()) {
      output->tensor = tensor;
      output->shape = utils::GetTensorShapeFromTensorShapeProto(*arg.Shape());
      break;
    }
  }
  tensors_.insert({arg.Name(), tensor});
  return tensor;
}

}  // namespace npu

}  // namespace vsi
}  // namespace onnxruntime
