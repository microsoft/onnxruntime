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

#include "core/providers/openvino/openvino_graph.h"

namespace openvino_ep {

void OpenVINONode::CreateConvLayer(
    std::shared_ptr<InferenceEngine::Builder::Network>& builder,
    InferenceEngine::Precision precision,
    std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
    std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {

  auto conv_layer =
      std::make_shared<InferenceEngine::Builder::ConvolutionLayer>(
          onnx_node_->Name());

  //
  // *** Set inputs ***
  //
  auto formal_params = onnx_node_->Op()->inputs();

  for (size_t i = 0; i < formal_params.size(); i++) {
    auto formal_name = formal_params[i].GetName();

    if (formal_name == "X") {

      // Set Input info
      std::shared_ptr<OpenVINONode> in_ov_node = nullptr;

      if (node_connects_to_graph_inputs_) {
        auto input_name = input_defs_[i]->Name();
        in_ov_node = openvino_io_map[input_name];
      } else {
        in_ov_node = onnx_openvino_map[&(input_edges_[0].GetNode())];
      }
      InferenceEngine::idx_t in_port = 0;
      input_connections_.push_back( { in_ov_node, in_port });

    } else if (formal_name == "W") {

      // Set weights info
      std::string W_name = input_defs_[i]->Name();
      InferenceEngine::SizeVector size;
      size.push_back(GetTensorElemCount(W_name));

      if(precision == InferenceEngine::Precision::FP32) {
      auto ptrWeights = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision, size,
              InferenceEngine::Layout::C),(float*)GetTensorData(W_name, precision));
      conv_layer->setWeights(ptrWeights);
      }
      else if(precision == InferenceEngine::Precision::FP16) {
      auto ptrWeights = InferenceEngine::make_shared_blob(
          InferenceEngine::TensorDesc(precision, size,
              InferenceEngine::Layout::C), (short*)GetTensorData(W_name, precision));
      conv_layer->setWeights(ptrWeights);
      }
      conv_layer->setOutDepth(GetDimsVector(W_name)[0]); // Number of kernels

    } else if (formal_name == "B") {

      if(input_defs_.size() <= i) {
        std::cout <<  "Conv : Bias is not present" << std::endl;
        continue;
      }
      // Set biases info
      std::string B_name = input_defs_[i]->Name();
      InferenceEngine::SizeVector size;
      size.push_back(GetTensorElemCount(B_name));

      if(precision == InferenceEngine::Precision::FP32) {
        auto ptrBiases = InferenceEngine::make_shared_blob(
            InferenceEngine::TensorDesc(precision, size,
                InferenceEngine::Layout::C),
            (float*) GetTensorData(B_name, precision));
        conv_layer->setBiases(ptrBiases);
      } else if (precision == InferenceEngine::Precision::FP16) {
        auto ptrBiases = InferenceEngine::make_shared_blob(
            InferenceEngine::TensorDesc(precision, size,
                InferenceEngine::Layout::C),
            (short*) GetTensorData(B_name, precision));
        conv_layer->setBiases(ptrBiases);
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
  auto attributes = onnx_node_->GetAttributes();

  // set dilations
  if(attributes.find("dilations") != attributes.end()) {
    auto dilations_ints = attributes["dilations"].ints();
    std::vector<size_t> dilations;
    for (size_t i = 0; i < dilations_ints.size(); i++) {
      dilations.push_back(size_t(dilations_ints[i]));
    }
    conv_layer->setDilation(dilations);
  }

  // set group
  if (attributes.find("group") != attributes.end()) {
    auto group = attributes["group"].i();
    conv_layer->setGroup(size_t(group));
  }

  // set strides
    auto strides_ints = attributes["strides"].ints();
    std::vector<size_t> strides;
    for (int i = 0; i < strides_ints.size(); i++) {
      strides.push_back(size_t(strides_ints[i]));
    }
    conv_layer->setStrides(strides);

  // set padding
  if (attributes.find("auto_pad") != attributes.end()) {
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
        throw "Conv layer: paddings SAME_UPPER not implemented";

      } else if (auto_pad == "SAME_LOWER") {
        // TODO: fill these
        throw "Conv layer: paddings SAME_LOWER not implemented";

      }
    }
    conv_layer->setPaddingsBegin(pad_begins);
    conv_layer->setPaddingsEnd(pad_ends);
  }

  // set kernel shape
  if (attributes.find("kernel_shape") != attributes.end()) {
    auto kernel_shape_ints = attributes["kernel_shape"].ints();
    std::vector<size_t> kernel_shape;
    for (int i = 0; i < kernel_shape_ints.size(); i++) {
      kernel_shape.push_back(size_t(kernel_shape_ints[i]));
    }
    conv_layer->setKernel(kernel_shape);
  }

  layerID_ = builder->addLayer(*conv_layer);

  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(conv_layer);
}
} // namespce openvino_ep
