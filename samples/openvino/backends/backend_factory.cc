// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "../contexts.h"
#include "../ibackend.h"
#include "basic_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend>
BackendFactory::MakeBackend(void* model_proto,
                            size_t model_proto_len,
                            GlobalContext& global_context,
                            const SubGraphContext& subgraph_context,
                            EPCtxHandler& ep_ctx_handle) {
  std::string type = global_context.device_type;
  if (type == "CPU" || type.find("GPU") != std::string::npos ||
      type.find("NPU") != std::string::npos ||
      type.find("HETERO") != std::string::npos ||
      type.find("MULTI") != std::string::npos ||
      type.find("AUTO") != std::string::npos) {
    std::shared_ptr<IBackend> concrete_backend_;
    try {
      concrete_backend_ = std::make_shared<BasicBackend>(model_proto, model_proto_len, global_context, subgraph_context, ep_ctx_handle);
    } catch (std::string const& msg) {
      throw std::runtime_error(msg);
    }
    return concrete_backend_;
  }
  throw std::runtime_error("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
}
}  // namespace openvino_ep
}  // namespace onnxruntime
