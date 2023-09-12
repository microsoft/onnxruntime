// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "onnx_converter.h"
#include "shl_common.h"

using std::string;
using std::vector;

namespace onnxruntime {
namespace shl_ep {

#define HAS(map, key) \
  (map.find(key) != map.end())

void OnnxToShlConverter::InitAllTensor(const GraphViewer& graph_viewer) {
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  const std::vector<const NodeArg*>& all_nodes = graph_viewer.GetInputsIncludingInitializers();

  // input and constant
  for (const auto* node : all_nodes) {
    csinn_tensor* shl_tensor = shl_ep::CreateShlTensor(node, session_);
    shl_tensor_map[node->Name()] = shl_tensor;
  }

  // per layer output
  for (const auto& layer : graph_viewer.Nodes()) {
    for (const auto output : layer.OutputDefs()) {
      csinn_tensor* shl_tensor = shl_ep::CreateShlTensor(output, session_);
      shl_tensor_map[output->Name()] = shl_tensor;
    }
  }

  const auto& input_tensors = graph_viewer.GetInputs();
  const auto& output_tensors = graph_viewer.GetOutputs();

  // set constant buf
  for (const auto& tensor : init_tensors) {
    Initializer unpacked_tensor(*(tensor.second));
    const uint8_t* data_buf = unpacked_tensor.DataAsByteSpan().data();
    csinn_tensor* shl_tensor = shl_tensor_map.at(tensor.first);
    int size = csinn_tensor_byte_size(shl_tensor);
    shl_tensor->data = shl_mem_alloc(size);
    shl_tensor->layout = GetShlWeightLayoutEnum(shl_tensor->dim_count);
    memcpy(shl_tensor->data, data_buf, size);
    shl_tensor->is_const = 1;
  }

  csinn_set_input_number(input_tensors.size(), session_);
  csinn_set_output_number(output_tensors.size(), session_);

  auto set_sess_dynamic_shape = [session = session_](csinn_tensor* t) {
    for (int i = 0; i < t->dim_count; i++) {
      if (t->dim[i] < 0) {
        if (t->dim[i] == -1) {
          session->dynamic_shape = true;
          break;
        } else {
          throw std::invalid_argument("Error obtaining shape value.");
        }
      }
    }
  };

  for (uint i = 0; i < input_tensors.size(); i++) {
    auto tensor = input_tensors[i];
    auto shl_tensor = shl_tensor_map.at(tensor->Name());
    set_sess_dynamic_shape(shl_tensor);
    csinn_set_tensor_entry(shl_tensor, session_);
    csinn_set_input(i, shl_tensor, session_);
  }
}

void OnnxToShlConverter::Convert(const GraphViewer& graph_viewer) {
  // 1.  create all shl tensor and set constant data
  InitAllTensor(graph_viewer);

  // 2. set attr and build shl graph
  marked_fusible_map = shl_ep::MarkfusibleNodes(graph_viewer);
  all_fusible_nodes = shl_ep::GetAllFusionNode(marked_fusible_map);
  bool clear_buffer = true;
  for (const auto& node : graph_viewer.Nodes()) {
    NodeConvert(node, clear_buffer);
  }

  // 3. set output tensor
  const auto& output_tensors = graph_viewer.GetOutputs();
  for (uint i = 0; i < output_tensors.size(); i++) {
    auto tensor_name = output_tensors[i];
    auto shl_tensor = shl_tensor_map[tensor_name->Name()];
    csinn_set_output(i, shl_tensor, session_);
  }
  csinn_session_setup(session_);
}

void OnnxToShlConverter::Conv2D(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_conv2d_params>(session_);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  const auto dilations = helper.Get("dilations", std::vector<int32_t>{1, 1});

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->group = helper.Get("group", 1);
  params->stride_height = strides[0];
  params->stride_width = strides[1];
  params->pad_top = pads[0];
  params->pad_left = pads[1];
  params->pad_down = pads[2];
  params->pad_right = pads[3];
  params->dilation_height = dilations[0];
  params->dilation_width = dilations[1];

  std::string bias;
  if (node.InputDefs().size() >= 3) {
    bias = node.InputDefs()[2]->Name();
  }

  std::string input = node.InputDefs()[0]->Name();
  std::string weight = node.InputDefs()[1]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto weight_tensor = shl_tensor_map.at(weight);
  auto bias_tensor = bias.size() ? shl_tensor_map.at(bias) : csinn_alloc_tensor(session_);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  csinn_conv2d_init(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
  csinn_conv2d(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
}

void OnnxToShlConverter::Conv(const onnxruntime::Node& node) {
  auto input = node.InputDefs()[0];
  switch (shl_tensor_map[input->Name()]->dim_count) {
    case 3:
      Conv1D(node);
      break;
    case 4:
      Conv2D(node);
      break;
    case 5:
      Conv3D(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of Conv.");
      ;
  }
}

void OnnxToShlConverter::Gemm(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_fc_params>(session_);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string bias;
  if (node.InputDefs().size() >= 3) {
    bias = node.InputDefs()[2]->Name();
  }

  std::string input = node.InputDefs()[0]->Name();
  std::string weight = node.InputDefs()[1]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto weight_tensor = shl_tensor_map.at(weight);
  auto bias_tensor = bias.size() ? shl_tensor_map.at(bias) : csinn_alloc_tensor(session_);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  params->units = weight_tensor->dim[0];

  csinn_fullyconnected_init(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
  csinn_fullyconnected(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
}

}  // namespace shl_ep
}  // namespace onnxruntime
