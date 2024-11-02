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
        OpenVINOExecutionProvider* p = static_cast<OpenVINOExecutionProvider*>(this_);
        this_->extra_param_for_create_state_func = p;
        for (int i = 0; i < cnt; i++) {
            p->global_context_->use_api_2 = true;

            // During backend creation, we check if user wants to use precompiled blob onnx model or the original model
            // For precompiled blob, directly load the model instead of compiling the model
            // For original model, check if the user wants to export a model with pre-compiled blob

            std::unique_ptr<openvino_ep::BackendManager> backend_manager =
                std::make_unique<openvino_ep::BackendManager>(*p->global_context_,
                                                            node[i],
                                                            graph[i],
                                                            p->ep_ctx_handle_);

            if (p->global_context_->export_ep_ctx_blob && !p->ep_ctx_handle_.IsValidOVEPCtxGraph()) {
                backend_manager->ExportCompiledBlobAsEPCtxNode(graph[i]);
            }
            const char* fused_node_name = nullptr;
            graph_api_->OrtNode_GetName(node[i], &fused_node_name);
            p->backend_managers_.emplace(fused_node_name, std::move(backend_manager));

            node_compute_info[i].CreateFunctionStateFunc = [](OrtComputeContext* context, void* extra_param, void** state) -> int {
                OpenVINOExecutionProvider* this_ = reinterpret_cast<OpenVINOExecutionProvider*>(extra_param);
                std::unique_ptr<OpenVINOEPFunctionState> p = std::make_unique<OpenVINOEPFunctionState>();
                p->AllocateFunc = context->AllocateFunc;
                p->DestroyFunc = context->DestroyFunc;
                p->allocator_handle = context->allocator_handle;
                p->node_name = context->node_name;
                p->backend_manager = this_->backend_managers_[context->node_name].get();
                *state = p.release();
                return 0;
            };
            node_compute_info[i].ComputeFunc = [](void* state, void* extra_param, const OrtApi* api, OrtKernelContext* context) -> OrtStatusPtr {
                auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
                try {
                    function_state->backend_manager->Compute(context);
                } catch (const std::exception& ex) {
                    return api_->CreateStatus(OrtErrorCode::ORT_EP_FAIL, ex.what());
                }
                return nullptr;
            };
            node_compute_info[i].DestroyFunctionStateFunc = [](void* state) {
                if (state) {
                    OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
                    delete function_state;
                }
            };
        }
        return nullptr;
    };

    OrtExecutionProvider::ReleaseIndexedSubGraphs = [](OrtIndexedSubGraph** indexed_sub_graphs, size_t num_sub_graph) {
      if (indexed_sub_graphs == nullptr) return;
      for (size_t i = 0; i < num_sub_graph; i++) {
        OrtIndexedSubGraph* sub_graph = indexed_sub_graphs[i];
        delete[] sub_graph->node_index;
        delete sub_graph->meta_def;
        delete sub_graph;
      }
      delete[] indexed_sub_graphs;
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
