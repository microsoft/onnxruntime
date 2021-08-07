// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "basic_backend.h"
#include "vadm_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend>
BackendFactory::MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                            GlobalContext& global_context,
                            const SubGraphContext& subgraph_context) {
  std::string type = global_context.device_type;
  if (type.find("HDDL") != std::string::npos) {
    return std::make_shared<VADMBackend>(model_proto, global_context, subgraph_context);
  } else if (type == "CPU" || type == "GPU" || type == "MYRIAD" || type.find("HETERO") != std::string::npos || type.find("MULTI") != std::string::npos) {
    return std::make_shared<BasicBackend>(model_proto, global_context, subgraph_context);
  } else {
    ORT_THROW("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
