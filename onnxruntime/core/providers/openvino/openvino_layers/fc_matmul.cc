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

void OpenVINONode::CreateFCMatMulLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {

  auto fc_matmul_layer =
      std::make_shared<InferenceEngine::Builder::FullyConnectedLayer>(
          onnx_node_->Name());

    size_t output_size = 1;

  //
  // *** Set inputs ***
  //
  auto formal_params = onnx_node_->Op()->inputs();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();

    if (formal_name == "A") {

      // Set Input info
      std::shared_ptr<OpenVINONode> in_ov_node = nullptr;

      auto input_name = input_defs_[i]->Name();
      auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(input_defs_[i]->Shape()));
      output_size *= shape_vector[0];
      if (node_connects_to_graph_inputs_) {
        in_ov_node = openvino_io_map[input_name];
      } else {
        in_ov_node = onnx_openvino_map[&(input_edges_[0].GetNode())];
      }
      InferenceEngine::idx_t in_port = 0;
      input_connections_.push_back( { in_ov_node, in_port });

    } else if (formal_name == "B") {

      // Set weights info
      std::string W_name = input_defs_[i]->Name();
      InferenceEngine::SizeVector size;
      size.push_back(GetTensorElemCount(W_name));
      auto ptrWeights = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision, size,
              InferenceEngine::Layout::C), (float*)GetTensorData(W_name,precision));
      fc_matmul_layer->setWeights(ptrWeights);
      auto dims = GetDimsVector(W_name);
      for(int i = 1; i < dims.size(); i++){
          output_size *= dims[i];
      }
    } else {
      std::stringstream msg;
      msg << "Node: " << onnx_node_->Name() << "| Param: "
          << formal_name.c_str() << "not found";
      throw msg.str();
    }
  }

  //
  // *** Set Outputs ***
  //
  formal_params = onnx_node_->Op()->outputs();
  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();
    if (formal_name == "Y") {

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
      throw msg.str();
    }
  }

  //
  // *** Set attributes ***
  //
  fc_matmul_layer->setOutputNum(output_size);

  layerID_ = builder->addLayer(*fc_matmul_layer);
}
} // namespce openvino_ep
