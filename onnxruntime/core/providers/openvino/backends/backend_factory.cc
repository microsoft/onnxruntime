// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/ibackend.h"
#include "core/common/common.h"
#include "basic_backend.h"
#include "vadm_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend>
BackendFactory::MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                            GlobalContext& global_context,
                            const SubGraphContext& subgraph_context) {
  std::string type = subgraph_context.device_id;
  if (type == "CPU" || type == "GPU" || type == "MYRIAD" || type == "HETERO:FPGA,CPU") {
    return std::make_shared<BasicBackend>(model_proto, global_context, subgraph_context);
  } else if (type == "HDDL") {
    return std::make_shared<VADMBackend>(model_proto, global_context, subgraph_context);
  } else {
    ORT_THROW("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime