// Copyright (C) Intel Corporation
// Licensed under the MIT License
#include <filesystem>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "core/providers/openvino/qdq_transformations/qdq_stripping.h"
#include "core/providers/openvino/exceptions.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "openvino/core/version.hpp"
#ifdef USE_OVEP_NPU_MEMORY
#include "core/providers/openvino/ov_allocator.h"
#endif
#include "ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

std::atomic<uint32_t> OpenVINOExecutionProvider::global_session_counter_{0};

// Parking this code here for now before it's moved to the factory
#if defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
static std::vector<std::string> parseDevices(const std::string& device_string,
                                             const std::vector<std::string>& available_devices) {
  std::string comma_separated_devices = device_string;
  if (comma_separated_devices.find(":") != std::string::npos) {
    comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
  }
  auto devices = split(comma_separated_devices, ',');
  if (devices.size() < 2) {
    print_build_options();
    ORT_THROW("Invalid device string: " + device_string);
  }
  std::set<std::string> dev_options = {"CPU", "GPU", "NPU"};

  for (auto& device : available_devices) {
    if (dev_options.find(device) == dev_options.end()) {
      auto dev_options_update = dev_options.emplace(device);
    }
  }

  for (const std::string& dev : devices) {
    if (!std::count(dev_options.begin(), dev_options.end(), dev)) {
      print_build_options();
      ORT_THROW("Invalid device string: " + device_string);
    }
  }
  return devices;
}
#endif

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const ProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider},
      session_context_(info),
      ov_core_(OVCore::Get()),
      shared_context_manager_(SharedContextManager::Get()),
      ep_ctx_handle_{session_context_.openvino_sdk_version, *GetLogger(), shared_context_manager_} {
  InitProviderOrtApi();
#ifdef _WIN32
  session_id_ = global_session_counter_.fetch_add(1) + 1;
  // Trace all runtime options (includes both session and provider options)
  OVTracing::Instance().LogAllRuntimeOptions(session_id_, session_context_);
#endif
}

OpenVINOExecutionProvider::~OpenVINOExecutionProvider() {
  for (auto& backend_manager : backend_managers_) {
    backend_manager.ShutdownBackendManager();
  }
  backend_managers_.clear();
}

std::vector<std::unique_ptr<ComputeCapability>>
OpenVINOExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                         const IKernelLookup& /*kernel_lookup*/,
                                         const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                         IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Enable CI Logs
  if (!(GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG").empty())) {
    std::cout << "In the OpenVINO EP" << std::endl;
  }
  openvino_ep::GetCapability obj(ep_ctx_handle_,
                                 graph_viewer,
                                 session_context_.device_type,
                                 session_context_.enable_qdq_optimizer);
  result = obj.Execute();
  session_context_.is_wholly_supported_graph = obj.IsWhollySupportedGraph();
  session_context_.has_external_weights = obj.HasExternalWeights();
  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  auto& logger = *GetLogger();
  Status status = Status::OK();

  try {
    if (session_context_.so_context_enable && session_context_.so_context_embed_mode && session_context_.so_share_ep_contexts) {
      return Status(common::StatusCategory::ONNXRUNTIME, common::EP_FAIL,
                    std::string("Invalid EP context configuration: ") + kOrtSessionOptionEpContextEmbedMode + " must be 0 if " + kOrtSessionOptionShareEpContexts + " is 1.");
    }

    if (!fused_nodes.empty()) {
      // Assume these properties are constant for all the model subgraphs, otherwise move to SubGraphContext
      const auto& graph_body_viewer_0 = fused_nodes[0].filtered_graph.get();
      session_context_.onnx_model_path_name = graph_body_viewer_0.ModelPath().string();
      session_context_.onnx_opset_version =
          graph_body_viewer_0.DomainToVersionMap().at(kOnnxDomain);
    }

    shared_context_ = ep_ctx_handle_.Initialize(fused_nodes, session_context_);
    ORT_ENFORCE(shared_context_,
                "Failed to create or retrieve SharedContext");

    struct OpenVINOEPFunctionState {
      AllocateFunc allocate_func = nullptr;
      DestroyFunc destroy_func = nullptr;
      AllocatorHandle allocator_handle = nullptr;
      BackendManager& backend_manager;
    };

    for (const FusedNodeAndGraph& fused_node_graph : fused_nodes) {
      const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
      const Node& fused_node = fused_node_graph.fused_node;

      NodeComputeInfo compute_info;

      // During backend creation, we check if user wants to use precompiled blob onnx model or the original model
      // For precompiled blob, directly load the model instead of compiling the model
      // For original model, check if the user wants to export a model with pre-compiled blob

      auto& backend_manager = backend_managers_.emplace_back(session_context_,
                                                             *shared_context_,
                                                             fused_node,
                                                             graph_body_viewer,
                                                             logger,
                                                             ep_ctx_handle_);
      compute_info.create_state_func =
          [&backend_manager](ComputeContext* context, FunctionState* state) {
            OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState{
                .allocate_func = context->allocate_func,
                .destroy_func = context->release_func,
                .allocator_handle = context->allocator_handle,
                .backend_manager = backend_manager};
            *state = static_cast<FunctionState>(p);
            return 0;
          };

      compute_info.compute_func = [](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
        auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
        try {
          function_state->backend_manager.Compute(context);
        } catch (const std::exception& ex) {
          return common::Status(common::ONNXRUNTIME, common::FAIL, ex.what());
        }
        return Status::OK();
      };

      compute_info.release_state_func =
          [](FunctionState state) {
            if (state) {
              OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
              delete function_state;
            }
          };

      node_compute_funcs.push_back(std::move(compute_info));
    }

    // Export compiled blobs as EPContext nodes if context enable is set
    if (session_context_.so_context_enable) {
      auto backend_it = backend_managers_.begin();
      bool is_first = true;

      for (const auto& fused_node_graph : fused_nodes) {
        const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;

        // Set include_embed_data to true only for the first backend manager
        backend_it->TryExportCompiledBlobAsEPCtxNode(graph_body_viewer, is_first);

        is_first = false;
        ++backend_it;
      }

      // bit clunky ideally we should try to fold this into ep context handler
      if (!session_context_.so_context_embed_mode) {
        shared_context_->Serialize();
        if (session_context_.so_stop_share_ep_contexts) {
          shared_context_manager_->ClearActiveSharedContext();
        }
      }
    }
  } catch (const ovep_exception& ex) {
    status = ex;
  }

  return status;
}

#ifdef USE_OVEP_NPU_MEMORY
std::vector<AllocatorPtr> OpenVINOExecutionProvider::CreatePreferredAllocators() {
  if (session_context_.device_type.find("NPU") != std::string::npos) {
    AllocatorCreationInfo npu_allocator_info{
        [this](OrtDevice::DeviceId device_id) {
          return std::make_unique<OVRTAllocator>(
              OVCore::Get()->core,
              OrtDevice::NPU,
              device_id,
              OpenVINO_RT_NPU);
        },
        0,
    };

    // fill in allocator
    return std::vector<AllocatorPtr>{CreateAllocator(npu_allocator_info)};
  } else {
    return std::vector<AllocatorPtr>{};
  }
}
#endif

common::Status OpenVINOExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                              gsl::span<const char* const> values) {
  std::string workload_type = "";
  // Ensure the number of keys and values match
  if (keys.size() != values.size()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Mismatched keys and values sizes.");
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string key = keys[i];
    std::string value = values[i];

    if (key == kOrtEpDynamicOptionsWorkloadType) {
      if (value == "Efficient") {
        workload_type = "EFFICIENT";
      } else if (value == "Default") {
        workload_type = "DEFAULT";
      } else {
        LOGS_DEFAULT(WARNING) << "Unknown workload_type - ignoring " << key << "/" << value;
        LOGS_DEFAULT(WARNING) << "Supported types are 'Efficient' and 'Default' \n";
      }
      if (workload_type != "") {
        LOGS_DEFAULT(VERBOSE) << "SetEpDynamicOptions - modifying: " << key << "/" << value;
        for (auto& backend : backend_managers_) {
          ov::CompiledModel ov_compiled_model = backend.GetOVCompiledModel();
          if (ov_compiled_model) {
            ov_compiled_model.set_property(ov::workload_type(workload_type));
          } else {
            LOGS_DEFAULT(VERBOSE) << "Model is not compiled in OV as its dynamic";
            ov::AnyMap map;
            map["WORKLOAD_TYPE"] = workload_type;
            if (session_context_.device_type == "NPU")
              session_context_.load_config["NPU"] = std::move(map);
            else
              ORT_THROW(" WORKLOAD_TYPE property is supported only for NPU");
          }
        }
      }
    } else if (key == "kvcache_rewind") {
      // Convert kvcache_rewind value to int64_t
      int64_t index;
      try {
        index = std::stoll(value);
      } catch (const std::exception& e) {
        LOGS_DEFAULT(WARNING) << "Conversion for kvcache_rewind string value to int64_t index failed."
                              << "Exception:" + std::string(e.what());
        return Status::OK();
      }

      // Trigger KVCache Rewind for target Backend
      for (auto& backend : backend_managers_) {
        if (index >= 0) {
          backend.RewindKVCache(static_cast<size_t>(index));
        } else {
          LOGS_DEFAULT(WARNING) << "kvcache_rewind index is < 0:\t" << index;
        }
      }
    } else {
      // Handle unknown options
      LOGS_DEFAULT(WARNING) << "Unknown key/value pair - ignoring " << key << "/" << value;
    }
  }
  return Status::OK();
}

const InlinedVector<const Node*> OpenVINOExecutionProvider::GetEpContextNodes() const {
  return ep_ctx_handle_.GetEPCtxNodes();
}

}  // namespace openvino_ep
}  // namespace onnxruntime
