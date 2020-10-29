#pragma once

#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace openvino_ep {

#if (defined OPENVINO_2020_2) || (defined OPENVINO_2020_3)
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2020_2(const onnxruntime::GraphViewer& graph_viewer, const std::string device_type);

#elif defined OPENVINO_2020_4
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2020_4(const onnxruntime::GraphViewer& graph_viewer, const std::string device_type);

#elif defined OPENVINO_2021_1
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2021_1(const onnxruntime::GraphViewer& graph_viewer, const std::string device_id);

#endif

} //namespace openvino_ep
} //namespace onnxruntime