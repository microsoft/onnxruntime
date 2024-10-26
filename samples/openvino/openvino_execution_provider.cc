#include <cassert>
#include <filesystem>
#include "openvino_execution_provider.h"
#include "openvino_utils.h"
#include "ov_versions/capability.h"

namespace onnxruntime {

const OrtApi* OpenVINOExecutionProvider::api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const OrtGraphApi* OpenVINOExecutionProvider::graph_api_ = OpenVINOExecutionProvider::api_->GetGraphApi(ORT_API_VERSION);

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const char* ep_type, const ProviderOptions& provider_options) : OrtExecutionProvider() {
    OrtExecutionProvider::GetCapability = [](const OrtExecutionProvider* this_, const OrtGraphViewer* graph_viewer, size_t* cnt, OrtIndexedSubGraph*** indexed_sub_graph) {
        const OpenVINOExecutionProvider* p = static_cast<const OpenVINOExecutionProvider*>(this_);
        std::string openvino_sdk_version = std::to_string(p->global_context_->OpenVINO_Version.at(0)) + "." +
                                            std::to_string(p->global_context_->OpenVINO_Version.at(1));

        // Check for valid ctx node and maintain state for validity
        if (p->ep_ctx_handle_.CheckForOVEPCtxNode(graph_viewer, openvino_sdk_version)) {
            int num_nodes = 0;
            graph_api_->OrtGraph_NumberOfNodes(graph_viewer, &num_nodes);
            assert((num_nodes==1) && "[Invalid Graph] EPContext Model with OpenVINO compiled blob should not have more than one node");
        }

        // Enable CI Logs
        if (!(GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG").empty())) {
            std::cout << "In the OpenVINO EP" << std::endl;
        }
        const void* model_path = nullptr;
        graph_api_->OrtGraph_GetModelPath(graph_viewer, &model_path);
        p->global_context_->onnx_model_path_name = reinterpret_cast<const std::filesystem::path*>(model_path)->string();

//        global_context_->onnx_opset_version =
//            graph_viewer.DomainToVersionMap().at(kOnnxDomain);

        p->global_context_->model_precision = [&](const OrtGraphViewer* graph_viewer) {
            // return empty if graph has no inputs or if types are not one of FP32/FP16
            // else assume the type of the first input
            const char** required_inputs = nullptr;
            size_t input_count = 0;
            graph_api_->OrtGraph_GetRequiredInputs(graph_viewer, &required_inputs, &input_count);
            if (input_count == 0) return "";
            if (p->global_context_->precision_str == "ACCURACY" &&
                p->global_context_->device_type.find("GPU") != std::string::npos) {
                OrtValueInfoRef* valueinfo = nullptr;
                graph_api_->OrtGraph_GetValueInfo(graph_viewer, required_inputs[0], &valueinfo);
                ONNXTensorElementDataType data_type = valueinfo->data_type;
                graph_api_->OrtGraph_ReleaseValueInfo(valueinfo);
                graph_api_->ReleaseCharArray(required_inputs);
                if (data_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return "FP32";
                if (data_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) return "FP16";
            }
            return "";
        }(graph_viewer);

        openvino_ep::GetCapability obj(graph_viewer,
                                        p->global_context_->device_type,
                                        p->global_context_->enable_qdq_optimizer);
        *cnt = obj.Execute(indexed_sub_graph);
        p->global_context_->is_wholly_supported_graph = obj.IsWhollySupportedGraph();
    };

    OrtExecutionProvider::Compile = [](OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo* node_compute_info) -> OrtStatusPtr {
        return nullptr;
    };
}

OpenVINOExecutionProviderFactory::OpenVINOExecutionProviderFactory() {
    OrtExecutionProviderFactory::CreateExecutionProvider = [](OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size) -> OrtExecutionProvider* {
        ProviderOptions options;
        for (size_t i = 0; i < option_size; i++) options[ep_option_keys[i]] = ep_option_values[i];
        std::unique_ptr<OpenVINOExecutionProvider> ret = std::make_unique<OpenVINOExecutionProvider>(OpenVINOEp.c_str(), std::move(options));
        return ret.release();
    };
}
}   // namespace onnxruntime

#ifdef __cplusplus
extern "C" {
#endif
OrtExecutionProviderFactory* RegisterCustomEp() {
    std::unique_ptr<onnxruntime::OpenVINOExecutionProviderFactory> ret = std::make_unique<onnxruntime::OpenVINOExecutionProviderFactory>();
    return ret.release();
}
#ifdef __cplusplus
}
#endif
