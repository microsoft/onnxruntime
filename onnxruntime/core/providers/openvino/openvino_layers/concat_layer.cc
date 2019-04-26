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

void OpenVINONode::CreateConcatLayer() {

  auto concat_layer =
      std::make_shared<InferenceEngine::Builder::ConcatLayer>(
          onnx_node_->Name());

  //
  // *** Set inputs ***
  //
  auto formal_params = onnx_node_->Op()->inputs();
  std::cout << "Formal params size is " << formal_params.size() << std::endl;


  for (size_t i = 0; i < formal_params.size(); i++) {

    auto formal_name = formal_params[i].GetName();

    if (formal_name == "inputs") {

    	// ????????????????
    	// Not sure why j is needed here.
    	//
        for(int j=0; j < onnx_node_->InputDefs().size(); j++){

           InferenceEngine::idx_t in_port = j;
           auto node_arg = onnx_node_->InputDefs()[j];
           input_connections_info_.insert({node_arg->Name(), in_port });
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
    if (formal_name == "concat_result") {

      InferenceEngine::idx_t out_port = i;
      auto out_arg = onnx_node_->OutputDefs()[i];
      output_connections_info_.insert({ out_arg->Name(), out_port });

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


  // set axis
  if(AttributeExists("axis")) {
    auto axis = attributes["axis"].i();
    concat_layer->setAxis(axis);
  }


  layerID_ = openvino_graph_->GetBuilder()->addLayer(*concat_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(concat_layer);
  std::cout << "Concat done " << std::endl;
}
} // namespce openvino_ep
