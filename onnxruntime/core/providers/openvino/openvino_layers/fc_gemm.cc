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


float* Get2DTransposedBuffer(float* src, size_t dimx, size_t dimy){
    float* dst = new float[dimx*dimy];
    for(size_t i = 0; i < dimx ; i++){
        for(size_t j = 0; j < dimy; j++){
            dst[j*dimx+i] = src[i*dimy +j];
        }
    }
    return dst;
}

void OpenVINONode::CreateFCGemmLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {

  auto fc_gemm_layer =
      std::make_shared<InferenceEngine::Builder::FullyConnectedLayer>(
          onnx_node_->Name());

    size_t output_size = 1;

  //
  // *** Set inputs ***
  //

  auto attributes = onnx_node_->GetAttributes();
  auto transB = attributes["transB"].i();
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

        std::string W_name = input_defs_[i]->Name();
        auto dims = GetDimsVector(W_name);
        std::cout << "Dims 0 " << dims[0] << std::endl;
        std::cout << "Dims 1 " << dims[1] << std::endl;
        auto *transposed_weights = Get2DTransposedBuffer((float*)GetTensorData(W_name,precision),dims[0],dims[1]);

        std::cout << "In here " << std::endl;

        if(transB){

            InferenceEngine::SizeVector size;
            size.push_back(GetTensorElemCount(W_name));
            auto ptrWeights = InferenceEngine::make_shared_blob(
            InferenceEngine::TensorDesc(precision, size,
              InferenceEngine::Layout::C), transposed_weights);
            fc_gemm_layer->setWeights(ptrWeights);
        }
        else{
            InferenceEngine::SizeVector size;
            size.push_back(GetTensorElemCount(W_name));
            auto ptrWeights = InferenceEngine::make_shared_blob(
            InferenceEngine::TensorDesc(precision, size,
              InferenceEngine::Layout::C), (float*)GetTensorData(W_name,precision));
            fc_gemm_layer->setWeights(ptrWeights);
        }

      // Set weights info
      if(transB){
          output_size *= dims[0];
      }else {
          output_size *= dims[1];
      }
    } else if( formal_name == "C") {
        std::string B_name = input_defs_[i]->Name();
        InferenceEngine::SizeVector size;
        size.push_back(GetTensorElemCount(B_name));
        auto ptrBiases = InferenceEngine::make_shared_blob(
            InferenceEngine::TensorDesc(precision,size,
                InferenceEngine::Layout::C), (float*)GetTensorData(B_name,precision));
        fc_gemm_layer->setBiases(ptrBiases);
    }

    else {
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
  fc_gemm_layer->setOutputNum(output_size);
  std::cout << "Output Size is " << output_size << std::endl;

  layerID_ = builder->addLayer(*fc_gemm_layer);
}
} // namespce openvino_ep
