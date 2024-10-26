// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once
#include <vector>
#include <string>
#include <memory>
//#include "core/providers/openvino/ov_versions/data_ops.h"

namespace onnxruntime {
namespace openvino_ep {

class GetCapability {
 private:
  const OrtGraphViewer* graph_viewer_;
  std::string device_type_;
//  DataOps* data_ops_;
  bool is_wholly_supported_graph_ = false;
  static const OrtGraphApi* graph_api_;

 public:
  GetCapability(const OrtGraphViewer* graph_viewer_param,
                const std::string device_type_param,
                const bool enable_qdq_optimizer);
  size_t Execute(OrtIndexedSubGraph***);
  bool IsWhollySupportedGraph() {
    return is_wholly_supported_graph_;
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime
