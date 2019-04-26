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

#include "core/providers/openvino/openvino_node.h"
#include "core/providers/openvino/openvino_graph.h"

namespace openvino_ep {

void OpenVINONode::CreateNormLayer() {

  auto norm_layer =
      std::make_shared<InferenceEngine::Builder::NormLayer>(
          onnx_node_->Name());

  //
  // *** Set inputs ***
  //
  auto formal_params = onnx_node_->Op()->inputs();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();

    if (formal_name == "X") {

      // Set Input info
      InferenceEngine::idx_t in_port = 0;
      auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
      input_connections_info_.insert({ in_tensor_name, in_port });

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
  auto attributes = onnx_node_->GetAttributes();

  // set alpha
  norm_layer->setAcrossMaps(true);
  auto alpha = attributes["alpha"].f();
  norm_layer->setAlpha(alpha);

  // set beta
  auto beta = attributes["beta"].f();
  norm_layer->setBeta(beta);

  // set bias
//   auto bias = attributes["bias"].f();
  // needs feedback
  //norm_layer->

  // set size
  auto size = attributes["size"].i();
  norm_layer->setSize(size);


  layerID_ = openvino_graph_->GetBuilder()->addLayer(*norm_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(norm_layer);
}
} // namespce openvino_ep
