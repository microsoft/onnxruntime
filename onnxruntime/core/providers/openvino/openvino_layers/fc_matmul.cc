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

void OpenVINONode::CreateFCMatMulLayer() {

  auto fc_matmul_layer =
      std::make_shared<InferenceEngine::Builder::FullyConnectedLayer>(
          onnx_node_->Name());

    size_t output_size = 1;
    auto precision = openvino_graph_->precision_;

  //
  // *** Set inputs ***
  //
  auto formal_params = onnx_node_->Op()->inputs();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();

    if (formal_name == "A") {

      // Set Input info
      std::shared_ptr<OpenVINONode> in_ov_node = nullptr;

      // ?????
      // Where is this code being used?
      auto input_name = onnx_node_->InputDefs()[i]->Name();
      auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(onnx_node_->InputDefs()[i]->Shape()));
      output_size *= shape_vector[0];

      InferenceEngine::idx_t in_port = 0;
      auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
      input_connections_info_.insert({ in_tensor_name, in_port });

    } else if (formal_name == "B") {

      // Set weights info
      std::string W_name = onnx_node_->InputDefs()[i]->Name();
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

      InferenceEngine::idx_t out_port = 0;
      auto out_tensor_name = onnx_node_->OutputDefs()[i]->Name();
      output_connections_info_.insert({ out_tensor_name, out_port });

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

  layerID_ = openvino_graph_->GetBuilder()->addLayer(*fc_matmul_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(fc_matmul_layer);
}
} // namespce openvino_ep
