// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/providers/openvino/backends/basic_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend>
BackendFactory::MakeBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
                            SessionContext& session_context,
                            const SubGraphContext& subgraph_context,
                            SharedContext& shared_context,
                            ptr_stream_t& model_stream) {
  std::string type = session_context.device_type;
  if (type == "CPU" || type.find("GPU") != std::string::npos ||
      type.find("NPU") != std::string::npos ||
      type.find("HETERO") != std::string::npos ||
      type.find("MULTI") != std::string::npos ||
      type.find("AUTO") != std::string::npos) {
    std::shared_ptr<IBackend> concrete_backend_;
    try {
      concrete_backend_ = std::make_shared<BasicBackend>(model_proto, session_context, subgraph_context, shared_context, model_stream);
    } catch (std::string const& msg) {
      ORT_THROW(msg);
    }
    return concrete_backend_;
  } else {
    ORT_THROW("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
