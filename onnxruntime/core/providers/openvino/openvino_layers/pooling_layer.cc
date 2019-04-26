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

void OpenVINONode::CreatePoolingLayer(int poolingType) {


    auto pooling_layer =
        std::make_shared<InferenceEngine::Builder::PoolingLayer>(onnx_node_->Name());

  //
  // *** Set inputs ***
  //
    auto formal_params = onnx_node_->Op()->inputs();

    for(size_t i = 0; i < formal_params.size(); i++){
        auto formal_name = formal_params[i].GetName();

        if(formal_name == "X"){

            InferenceEngine::idx_t in_port = 0;
            auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
            input_connections_info_.insert({ in_tensor_name, in_port});
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
        output_connections_info_.insert( { out_tensor_name, out_port });

        } else if (formal_name == "Indices"){
            // std::cout << "Indices" << std::endl;
        }
        else {
        std::stringstream msg;
        msg << "Node: " << onnx_node_->Name() << "| Param: " << formal_name
            << "not found";
        throw msg.str();
        }
    }

  //
  // *** Set attributes ***
  //
    if(poolingType != 3){

        auto attributes = onnx_node_->GetAttributes();

        auto strides_ints = attributes["strides"].ints();
        std::vector<size_t> strides;
        for (int i = 0; i < strides_ints.size(); i++) {
            strides.push_back(size_t(strides_ints[i]));
        }
        pooling_layer->setStrides(strides);

        auto auto_pad = attributes["auto_pad"].s();
        std::vector<size_t> pad_begins, pad_ends;
        int num_axes = strides_ints.size();

        if (auto_pad == "VALID") { // No padding
            for (int i = 0; i < num_axes; i++) {
            pad_begins.push_back(0);
            pad_ends.push_back(0);
            }
        } else {
            if (auto_pad == "NOTSET") {
            auto pad_ints = attributes["pads"].ints();
            int pads_size_mid = pad_ints.size() / 2;
            for (int i = 0; i < pads_size_mid; i++) {
                pad_begins.push_back(size_t(pad_ints[i]));
                pad_ends.push_back(size_t(pad_ints[i + pads_size_mid]));
            }

            } else if (auto_pad == "SAME_UPPER") {
            // TODO: fill these
            throw "Max Pool layer: paddings SAME_UPPER not implemented";

            } else if (auto_pad == "SAME_LOWER") {
            // TODO: fill these
            throw "Max Pool layer: paddings SAME_LOWER not implemented";

            }
        }
        pooling_layer->setPaddingsBegin(pad_begins);
        pooling_layer->setPaddingsEnd(pad_ends);


        auto kernel_shape_ints = attributes["kernel_shape"].ints();
        std::vector<size_t> kernel_shape;
        for (int i = 0; i < kernel_shape_ints.size(); i++) {
            kernel_shape.push_back(size_t(kernel_shape_ints[i]));
        }
        pooling_layer->setKernel(kernel_shape);

        auto ceil_mode = attributes["ceil_mode"].i();
        if(ceil_mode == 0){
            pooling_layer->setRoundingType(InferenceEngine::Builder::PoolingLayer::RoundingType::FLOOR);
        }else if (ceil_mode == 1){
            pooling_layer->setRoundingType(InferenceEngine::Builder::PoolingLayer::RoundingType::CEIL);
        }
        pooling_layer->setExcludePad(true);
    }

    if(poolingType == 1){
        pooling_layer->setPoolingType(InferenceEngine::Builder::PoolingLayer::PoolingType::MAX);
    }else{
        pooling_layer->setPoolingType(InferenceEngine::Builder::PoolingLayer::PoolingType::AVG);
    }

    layerID_ = openvino_graph_->GetBuilder()->addLayer(*pooling_layer);
    layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(pooling_layer);
}
}
