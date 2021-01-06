// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once

namespace onnxruntime {
namespace openvino_ep {

#if (defined OPENVINO_2020_2) || (defined OPENVINO_2020_3)
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2020_2(const GraphViewer& graph_viewer, const std::string device_type);

#elif defined OPENVINO_2020_4
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2020_4(const GraphViewer& graph_viewer, const std::string device_type);

#elif defined OPENVINO_2021_1
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2021_1(const GraphViewer& graph_viewer, const std::string device_id);

#elif defined OPENVINO_2021_2
std::vector<std::unique_ptr<ComputeCapability>>
GetCapability_2021_2(const GraphViewer& graph_viewer, const std::string device_id);

#endif

}  //namespace openvino_ep
}  //namespace onnxruntime
