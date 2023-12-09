// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "../contexts.h"
#include "../ibackend.h"
#include "basic_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend>
BackendFactory::MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                            GlobalContext& global_context,
                            const SubGraphContext& subgraph_context) {
  std::string type = global_context.device_type;
  if (type == "CPU" || type.find("GPU") != std::string::npos ||
      type.find("VPUX") != std::string::npos ||
      type.find("HETERO") != std::string::npos ||
      type.find("MULTI") != std::string::npos ||
      type.find("AUTO") != std::string::npos) {
    std::shared_ptr<IBackend> concrete_backend_;
    try {
      concrete_backend_ = std::make_shared<BasicBackend>(model_proto, global_context, subgraph_context);
    } catch (std::string const& msg) {
      throw msg;
    }
    return concrete_backend_;
  } else {
    throw std::string("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
  }
}
}  // namespace openvino_ep
}  // namespace onnxruntime
