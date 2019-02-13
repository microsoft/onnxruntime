#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/openvino/openvino_graph.h"

namespace openvino_ep {

void OpenVINONode::CreateScaleShiftImgLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {

        std::cout << "In scale shift layer" << std::endl;

  auto scale_shift_img_layer =
      std::make_shared<InferenceEngine::Builder::ScaleShiftLayer>(
          onnx_node_->Name());

  //
  // *** Set inputs ***
  //

  auto formal_params = onnx_node_->Op()->inputs();

  size_t num_channels = 0;

  auto attributes = onnx_node_->GetAttributes();
  auto scale_attr = attributes["scale"].f();
  auto bias_attr = attributes["bias"].floats();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    std::cout << "Formal name is " << formal_name << std::endl;

    if (formal_name == "input") {

      // Set Input info
      std::shared_ptr<OpenVINONode> in_ov_node = nullptr;

      auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(input_defs_[i]->Shape()));
      num_channels = shape_vector[1];
      std::cout << "Num of channels is " << num_channels << std::endl;

      if (node_connects_to_graph_inputs_) {
        auto input_name = input_defs_[i]->Name();
        in_ov_node = openvino_io_map[input_name];
      } else {
        in_ov_node = onnx_openvino_map[&(input_edges_[0].GetNode())];
      }
      InferenceEngine::idx_t in_port = 0;
      input_connections_.push_back( { in_ov_node, in_port });

    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: "
          << formal_name.c_str() << "not found";
      throw msg.str();

    }

  }

    float* weights_tensor = new float[num_channels];
    float* bias_tensor = new float[num_channels];

    for(int i=0; i < num_channels; i++){
        weights_tensor[i] = scale_attr;
        bias_tensor[i] = bias_attr[i];
    }
    auto ptrWeights = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision,{num_channels},
            InferenceEngine::Layout::C), weights_tensor);
    auto ptrBiases = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision,{num_channels},
            InferenceEngine::Layout::C), bias_tensor);
    scale_shift_img_layer->setWeights(ptrWeights);
    scale_shift_img_layer->setBiases(ptrBiases);
  //
  // *** Set Outputs ***
  //
  formal_params = onnx_node_->Op()->outputs();
  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    if (formal_name == "output") {
        std::cout << "Output is set " << std::endl;

      std::shared_ptr<OpenVINONode> out_ov_node = nullptr;
      if (node_connects_to_graph_outputs_) {
        auto output_name = output_defs_[i]->Name();
        out_ov_node = openvino_io_map[output_name];
      } else {
        out_ov_node = onnx_openvino_map[&(output_edges_[0].GetNode())];
      }
      InferenceEngine::idx_t out_port = 0;
      output_connections_.push_back( { out_ov_node, out_port });

    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: " << formal_name
          << "not found";
    //   throw msg.str();
    }
  }

  layerID_ = builder->addLayer(*scale_shift_img_layer);
}
} // namespce openvino_ep
