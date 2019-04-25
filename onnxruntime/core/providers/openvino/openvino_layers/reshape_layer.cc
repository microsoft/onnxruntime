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

void OpenVINONode::CreateReshapeLayer() {


		// Reshape layer
		auto reshape_layer = std::make_shared<
				InferenceEngine::Builder::ReshapeLayer>(onnx_node_->Name());

		// Set Inputs
		auto formal_params = onnx_node_->Op()->inputs();

		for (size_t i = 0; i < formal_params.size(); i++) {
		    auto formal_name = formal_params[i].GetName();

		    if (formal_name == "data") {
		      // Set inputs info
		      InferenceEngine::idx_t in_port = 0;
		      auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
		      input_connections_info_.insert( { in_tensor_name, in_port });

		    } else if (formal_name == "shape") {

		      std::string W_name = onnx_node_->InputDefs()[i]->Name();
		      auto num_dims = GetTensorElemCount(W_name);

		      const ONNX_NAMESPACE::TensorProto* proto = openvino_graph_->GetInitializedTensor(W_name);
		      std::vector<int> shape;
		      for(int n=0; n<num_dims; n++) {
		    	  shape.push_back(int(proto->int64_data(n)));
		      }
		      // sets attribute
		      reshape_layer->setDims(shape);

		    } else {
		          std::stringstream msg;
		          msg << "Node: " << onnx_node_->Name() << "| Param: "
		              << formal_name.c_str() << "not found";
		          throw msg.str();

		    }
		}

		// Set Outputs
		    formal_params = onnx_node_->Op()->outputs();
		    for (size_t i = 0; i < formal_params.size(); i++) {
		      auto formal_name = formal_params[i].GetName();
		      if (formal_name == "reshaped") {

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


		// Set Attributes
		reshape_layer->setDims( { 3, 50176 });
		layerID_ = openvino_graph_->GetBuilder()->addLayer(*reshape_layer);
		layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(reshape_layer);
}
}
