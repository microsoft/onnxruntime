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

#include "core/providers/openvino/openvino_node.h"
#include "core/providers/openvino/openvino_graph.h"

namespace openvino_ep {

void OpenVINONode::CreateScaleShiftImgLayer() {

        std::cout << "In scale shift layer" << std::endl;

  auto scale_shift_img_layer =
      std::make_shared<InferenceEngine::Builder::ScaleShiftLayer>(
          onnx_node_->Name());

  auto precision = openvino_graph_->precision_;


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


      auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(onnx_node_->InputDefs()[i]->Shape()));
      num_channels = shape_vector[1];
      std::cout << "Num of channels is " << num_channels << std::endl;

      // Set Input info
      InferenceEngine::idx_t in_port = 0;
      auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
      input_connections_info_.insert( { in_tensor_name, in_port });

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

      InferenceEngine::idx_t out_port = 0;
      auto out_tensor_name = onnx_node_->OutputDefs()[i]->Name();
      output_connections_info_.insert( { out_tensor_name, out_port });

    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: " << formal_name
          << "not found";
    //   throw msg.str();
    }
  }

  layerID_ = openvino_graph_->GetBuilder()->addLayer(*scale_shift_img_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(scale_shift_img_layer);
}
} // namespce openvino_ep
