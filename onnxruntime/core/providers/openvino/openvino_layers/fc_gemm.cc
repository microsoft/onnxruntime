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


float* Get2DTransposedBuffer(float* src, size_t dimx, size_t dimy){
    float* dst = new float[dimx*dimy];
    for(size_t i = 0; i < dimx ; i++){
        for(size_t j = 0; j < dimy; j++){
            dst[j*dimx+i] = src[i*dimy +j];
        }
    }
    return dst;
}

void OpenVINONode::CreateFCGemmLayer() {

  auto fc_gemm_layer =
      std::make_shared<InferenceEngine::Builder::FullyConnectedLayer>(
          onnx_node_->Name());

    size_t output_size = 1;

    auto precision = openvino_graph_->precision_;

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
      InferenceEngine::idx_t in_port = 0;
      auto in_tensor_name = onnx_node_->InputDefs()[i]->Name();
      input_connections_info_.insert( { in_tensor_name , in_port });

    } else if (formal_name == "B") {

        std::string W_name = onnx_node_->InputDefs()[i]->Name();
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
        std::string B_name = onnx_node_->InputDefs()[i]->Name();
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

      InferenceEngine::idx_t out_port = 0;
      auto out_tensor_name = onnx_node_->OutputDefs()[i]->Name();
      output_connections_info_.insert( { out_tensor_name, out_port });

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

  layerID_ = openvino_graph_->GetBuilder()->addLayer(*fc_gemm_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(fc_gemm_layer);
}
} // namespce openvino_ep
