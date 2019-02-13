#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>

#include <inference_engine.hpp>
#include <ie_builders.hpp>

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/openvino/openvino_graph.h"

namespace openvino_ep {

void OpenVINONode::CreateScaleShiftLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {

        std::cout << "In scale shift layer" << std::endl;

  auto scale_shift_layer =
      std::make_shared<InferenceEngine::Builder::ScaleShiftLayer>(
          onnx_node_->Name());

  //
  // *** Set inputs ***
  //
  float* scale_tensor = nullptr;
  float* mean_tensor = nullptr;
  float* var_tensor = nullptr;
  float* bias_tensor = nullptr;

  auto formal_params = onnx_node_->Op()->inputs();

  size_t num_channels = 0;
    auto attributes = onnx_node_->GetAttributes();
    auto epsilon = attributes["epsilon"].f();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    std::cout << "Formal name is " << formal_name << std::endl;

    if (formal_name == "X") {

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
      std::cout << "End of X" << std::endl;

    } else if (formal_name == "scale") {


      // Set weights info
      std::string W_name = input_defs_[i]->Name();
      scale_tensor = new float[num_channels];
      scale_tensor = (float*)GetTensorData(W_name,precision);

    } else if (formal_name == "mean") {


      // Set biases info
      std::string W_name = input_defs_[i]->Name();
      mean_tensor = new float[num_channels];
      mean_tensor = (float*)GetTensorData(W_name,precision);

    } else if (formal_name == "var") {

      std::string W_name = input_defs_[i]->Name();
      var_tensor = new float[num_channels];
      var_tensor = (float*)GetTensorData(W_name,precision);
    } else if (formal_name == "B") {

      std::string W_name = input_defs_[i]->Name();

      bias_tensor = new float[num_channels];
      bias_tensor = (float*)GetTensorData(W_name,precision);
    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: "
          << formal_name.c_str() << "not found";
      throw msg.str();

    }


  }

    float* new_scale = new float[num_channels];
    float* new_bias = new float[num_channels];

    for(int i=0; i < num_channels; i++){
        float den = var_tensor[i] + epsilon;
        float den_sqrt = sqrt(den);
        new_scale[i] = scale_tensor[i]/den_sqrt;
        std::cout << "New scale is " << new_scale[i] << std::endl;

        float num = scale_tensor[i] * mean_tensor[i] * -1;
        new_bias[i] = num/den_sqrt + bias_tensor[i];
        std::cout << "New bias is " << new_bias[i] << std::endl;
    }
    auto ptrWeights = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision,{num_channels},
            InferenceEngine::Layout::C), new_scale);
    auto ptrBiases = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision,{num_channels},
            InferenceEngine::Layout::C), new_bias);
    scale_shift_layer->setWeights(ptrWeights);
    scale_shift_layer->setBiases(ptrBiases);
  //
  // *** Set Outputs ***
  //
  formal_params = onnx_node_->Op()->outputs();
  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    if (formal_name == "Y") {
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

  layerID_ = builder->addLayer(*scale_shift_layer);
}
} // namespce openvino_ep
