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

void OpenVINONode::CreateReshapeLayer(
        std::shared_ptr<InferenceEngine::Builder::Network>& builder,
		InferenceEngine::Precision precision,
		std::map<const onnxruntime::Node*, std::shared_ptr<OpenVINONode>>& onnx_openvino_map,
		std::map<std::string, std::shared_ptr<OpenVINONode>>& openvino_io_map) {

		(void) precision;
		(void) onnx_openvino_map;
		(void) openvino_io_map;

		// Reshape layer
		auto reshape_layer = std::make_shared<
				InferenceEngine::Builder::ReshapeLayer>();

		// Set Inputs


		// Set Outputs


		// Set Attributes
		reshape_layer->setDims( { 3, 50176 });
		layerID_ = builder->addLayer(*reshape_layer);
}
}
