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


void OpenVINONode::CreateEltwiseLayer(
    int EltwiseType) { //1-SUM/ADD, 2-MUL

        std::cout << "In eltwise" << std::endl;

//   auto eltwise_layer =
    //   std::make_shared<InferenceEngine::Builder::EltwiseLayer>(
        //   onnx_node_->Name());

  auto eltwise_layer =
      std::make_shared<InferenceEngine::Builder::EltwiseLayer>(
          onnx_node_->Name());
  //
  // *** Set inputs ***
  //
  auto formal_params = onnx_node_->Op()->inputs();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();

    if (formal_name == "A") {

      // Set Input info

      InferenceEngine::idx_t in_port = 0;
      auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
      input_connections_info_.insert({ in_tensor_name, in_port });

    } else if (formal_name == "B") {

    	// ?????
    	// Can the second input be an initializer instead?

		InferenceEngine::idx_t in_port = 1;
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
    if (formal_name == "C") {

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

  if(EltwiseType == 1)
	  eltwise_layer->setEltwiseType(InferenceEngine::Builder::EltwiseLayer::EltwiseType::SUM);
  else if(EltwiseType == 2)
	  eltwise_layer->setEltwiseType(InferenceEngine::Builder::EltwiseLayer::EltwiseType::MUL);
  else {
  	  std::stringstream msg;
  	  msg << "Eltwise Type unrecognized ";
  	  throw msg.str();

      }

  // *** No Attributes ***

  layerID_ = openvino_graph_->GetBuilder()->addLayer(*eltwise_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(eltwise_layer);
}

} // namespce openvino_ep
