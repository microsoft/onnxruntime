// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/ibackend.h"
#include "core/common/common.h"
#include "basic_backend.h"
#include "vadm_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend>
BackendFactory::MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                            const std::vector<int>& input_indexes,
                            const std::unordered_map<std::string, int>& output_names,
                            std::string type, InferenceEngine::Precision precision,
                            InferenceEngine::Core& ie_core, std::string subgraph_name) {
    if(type == "CPU" || type == "GPU" || type == "MYRIAD") {
        return std::make_shared<BasicBackend>(model_proto, input_indexes, output_names,
                                              type, precision, ie_core,subgraph_name);
    } else if (type == "HDDL") {
        return std::make_shared<VADMBackend>(model_proto, input_indexes, output_names,
                                             type, precision, ie_core, subgraph_name);
    }
    else {
        ORT_THROW("[OpenVINO-EP] Backend factory error: Unknown backend type: " + type);
    }
}

} // namespace openvino_ep
} // namespace onnxruntime