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

void OpenVINONode::CreateInputLayer() {

  // Create OpenVINO Op
  auto input_layer = std::make_shared<InferenceEngine::Builder::InputLayer>(onnx_nodearg_->Name());
  auto shape_vector = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*(onnx_nodearg_->Shape()));
  InferenceEngine::SizeVector ie_shape(shape_vector.begin(), shape_vector.end());
  input_layer->setPort(InferenceEngine::Port(ie_shape));
  layerID_ = openvino_graph_->GetBuilder()->addLayer(*input_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(input_layer);

  // Set inputs connections

  // Set outputs connections
  output_connections_info_.insert({onnx_nodearg_->Name(), 0});
}

} // namespace openvino_ep
