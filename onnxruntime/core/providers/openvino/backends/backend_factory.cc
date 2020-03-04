// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/openvino/ibackend.h"
#include "basic_backend.h"
#include "vadm_backend.h"

namespace onnxruntime {
namespace openvino_ep {

std::shared_ptr<IBackend> BackendFactory::MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto, std::vector<int> input_indexes, std::unordered_map<std::string, int> output_names, std::string type, InferenceEngine::Precision precision) {
    if(type == "CPU" || type == "GPU" || type == "MYRIAD") {
        return std::make_shared<BasicBackend>(model_proto, input_indexes, output_names, type, precision);
    } else if (type == "HDDL") {
        return std::make_shared<VADMBackend>(model_proto, input_indexes, output_names, type, precision);
    }
    else return nullptr;
}

} // namespace openvino_ep
} // namespace onnxruntime