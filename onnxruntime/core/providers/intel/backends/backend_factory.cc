// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <memory>
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/intel/ibackend.h"
#include "basic_backend.h"
#include "vadr_backend.h"

namespace onnxruntime {
namespace intel_ep {

std::shared_ptr<IBackend> BackendFactory::MakeBackend(const ONNX_NAMESPACE::ModelProto& model_proto, std::vector<int> input_indexes, std::string type, InferenceEngine::Precision precision) {
    if(type == "CPU") return std::make_shared<BasicBackend>(model_proto, input_indexes, type, precision);
    else return nullptr;
}

} // namespace intel_ep
} // namespace onnxruntime