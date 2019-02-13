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

void OpenVINONode::CreateUnsqueezeLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {


    // auto unsqueeze_layer =
        // std::make_shared<InferenceEngine::Builder::ReshapeLayer>(onnx_node_->Name());
    auto const_layer = std::make_shared<InferenceEngine::Builder::ConstLayer>(onnx_node_->Name());

  //
  // *** Set inputs ***
        // size_t layerID = -1;
        // InferenceEngine::idx_t src_port_idx = 0;
  //
    auto formal_params = onnx_node_->Op()->inputs();
    // std::vector<int64_t> size_vector;

    for(size_t i = 0; i < formal_params.size(); i++){
        auto formal_name = formal_params[i].GetName();

        if(formal_name == "data"){


            auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(input_defs_[i]->Shape()));
            // shape_vector.push_back(1);
            // shape_vector.push_back(1);
            // InferenceEngine::SizeVector size_vector(shape_vector.begin(),shape_vector.end());
            // std::cout << "Size vector size is " << size_vector.size() << std::endl;
            // for(int i = 0; i < size_vector.size(); i++){
//
                // std::cout << "Size vector is " << size_vector[i] << std::endl;
            // }

            std::string B_name = input_defs_[i]->Name();
            InferenceEngine::SizeVector size;
            size.push_back(GetTensorElemCount(B_name));

            auto ptrBiases = InferenceEngine::make_shared_blob(
                InferenceEngine::TensorDesc(precision, size,
                    InferenceEngine::Layout::C), (float*)GetTensorData(B_name,precision));

            // const_layer->setPort(InferenceEngine::Port(size_vector));
            const_layer->setData(ptrBiases);
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
    std::cout << "Formal name is  " << formal_params[0].GetName() << std::endl;
    for (size_t i = 0; i < formal_params.size(); i++) {
        auto formal_name = formal_params[i].GetName();
        if (formal_name == "expanded") {

            std::cout << "In output" << std::endl;

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

    // auto attributes = onnx_node_->GetAttributes();
    // std::cout << "Near attr" << std::endl;

    // std::cout << "Dims is " << size_vector.size() << std::endl;

    // auto axes = attributes["axes"].ints();
    // size_t final_size = size_vector.size();
    // for(auto axis : axes){
    //     if(axis > size_vector.size()-1){
    //         final_size++;
    //     }
    // }
    // std::cout << "Final size is " << final_size << std::endl;

    // std::vector<int> dims;
    // for(int i=0; i < final_size;i++){
    //     dims.push_back(0);
    // }
    // for(int i=0; i < axes.size(); i++){
    //     dims[axes[i]] = 1;
    // }

    // for(auto dim : dims){
    //     std::cout << "Dim is " << dim << std::endl;
    // }

    // std::vector<int> new_dims;
    // new_dims.push_back(0);

    // unsqueeze_layer->setDims(new_dims);


    layerID_ = builder->addLayer(*const_layer);
    // std::cout << "layerID_ " << layerID_ << std::endl;
    // std::cout << "layerID " << layerID << std::endl;
    // InferenceEngine::idx_t dst_port_idx = 0;

    // InferenceEngine::PortInfo src_port_info(layerID, src_port_idx);
    // InferenceEngine::PortInfo dst_port_info(layerID_, dst_port_idx);
    // builder->connect(src_port_info,dst_port_info);
    // other_layers_.push_back(const_layer);
    // layer_ = unsqueeze_layer;
}
}