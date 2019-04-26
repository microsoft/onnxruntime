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

void OpenVINONode::CreateOutputLayer() {

  auto output_layer = std::make_shared<InferenceEngine::Builder::OutputLayer>(onnx_nodearg_->Name());
  layerID_ = openvino_graph_->GetBuilder()->addLayer(*output_layer);
  layer_ = std::static_pointer_cast<InferenceEngine::Builder::LayerFragment>(output_layer);

  // Add inputs info
  input_connections_info_.insert({onnx_nodearg_->Name(), 0});

  // Add outputs info
}

} // namespace openvino_ep
