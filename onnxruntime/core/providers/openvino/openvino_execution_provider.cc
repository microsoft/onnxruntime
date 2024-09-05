// Copyright (C) Intel Corporation
// Licensed under the MIT License
#include <filesystem>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "openvino/core/version.hpp"
#ifdef USE_DEVICE_MEMORY
#include "core/providers/openvino/ov_allocator.h"
#endif

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const OpenVINOExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider} {
  InitProviderOrtApi();

  global_context_ = std::make_unique<openvino_ep::GlobalContext>();
  global_context_->device_type = info.device_type_;
  global_context_->precision_str = info.precision_;
  global_context_->enable_npu_fast_compile = info.enable_npu_fast_compile_;
  global_context_->cache_dir = info.cache_dir_;
  global_context_->model_priority = info.model_priority_;
  global_context_->num_streams = info.num_streams_;
  global_context_->context = info.context_;
  global_context_->enable_opencl_throttling = info.enable_opencl_throttling_;
  global_context_->disable_dynamic_shapes = info.disable_dynamic_shapes_;
  global_context_->num_of_threads = info.num_of_threads_;
  global_context_->OpenVINO_Version = {OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR};
  global_context_->export_ep_ctx_blob = info.export_ep_ctx_blob_;
  global_context_->enable_qdq_optimizer = info.enable_qdq_optimizer_;
  global_context_->disable_cpu_fallback = info.disable_cpu_fallback_;
  global_context_->ep_context_embed_mode = info.so_epctx_embed_mode_;

  // to check if target device is available
  // using ie_core capability GetAvailableDevices to fetch list of devices plugged in
  if (info.cache_dir_.empty()) {
    bool device_found = false;
    std::vector<std::string> available_devices = global_context_->ie_core.GetAvailableDevices();
    // Checking for device_type configuration
    if (info.device_type_ != "") {
      if (info.device_type_.find("HETERO") != std::string::npos ||
          info.device_type_.find("MULTI") != std::string::npos ||
          info.device_type_.find("AUTO") != std::string::npos) {
        device_found = true;
      } else {
        for (const std::string& device : available_devices) {
          if (device.rfind(info.device_type_, 0) == 0) {
            if (info.device_type_.find("GPU") != std::string::npos && (info.precision_ == "FP32" ||
                                                                       info.precision_ == "FP16" ||
                                                                       info.precision_ == "ACCURACY")) {
              device_found = true;
              break;
            }
            if (info.device_type_ == "CPU" && (info.precision_ == "FP32")) {
              device_found = true;
              break;
            }
            if (info.device_type_.find("NPU") != std::string::npos) {
              device_found = true;
              break;
            }
          }
        }
      }
    }
    if (!device_found) {
      ORT_THROW("[ERROR] [OpenVINO] Specified device - " + info.device_type_ + " is not available");
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
OpenVINOExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                         const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  std::string openvino_sdk_version = std::to_string(global_context_->OpenVINO_Version.at(0)) + "." +
                                     std::to_string(global_context_->OpenVINO_Version.at(1));

  // Check for valid ctx node and maintain state for validity
  if (ep_ctx_handle_.CheckForOVEPCtxNode(graph_viewer, std::move(openvino_sdk_version)))
    ORT_ENFORCE(graph_viewer.NumberOfNodes() == 1,
                "[Invalid Graph] EPContext Model with OpenVINO compiled blob should not have more than one node.");

  // Enable CI Logs
  if (!(GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG").empty())) {
    std::cout << "In the OpenVINO EP" << std::endl;
  }
  global_context_->onnx_model_path_name = graph_viewer.ModelPath().string();

  global_context_->onnx_opset_version =
      graph_viewer.DomainToVersionMap().at(kOnnxDomain);

  global_context_->model_precision = [&](const GraphViewer& graph_viewer) {
    // return empty if graph has no inputs or if types are not one of FP32/FP16
    // else assume the type of the first input
    if (graph_viewer.GetInputs().empty()) {
      return "";
    } else {
      auto input_type = graph_viewer.GetInputs()[0]->TypeAsProto()->tensor_type().elem_type();
      if (global_context_->precision_str == "ACCURACY" &&
          global_context_->device_type.find("GPU") != std::string::npos) {
        if (input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
          return "FP32";
        } else if (input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
          return "FP16";
        }
      }
    }
    return "";
  }(graph_viewer);

  openvino_ep::GetCapability obj(graph_viewer,
                                 global_context_->device_type,
                                 global_context_->enable_qdq_optimizer);
  result = obj.Execute();

  global_context_->is_wholly_supported_graph = obj.IsWhollySupportedGraph();

  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const FusedNodeAndGraph& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    NodeComputeInfo compute_info;

    global_context_->use_api_2 = true;

    // During backend creation, we check if user wants to use precompiled blob onnx model or the original model
    // For precompiled blob, directly load the model instead of compiling the model
    // For original model, check if the user wants to export a model with pre-compiled blob

    std::shared_ptr<openvino_ep::BackendManager> backend_manager =
        std::make_shared<openvino_ep::BackendManager>(*global_context_,
                                                      fused_node,
                                                      graph_body_viewer,
                                                      *GetLogger(),
                                                      ep_ctx_handle_);

    compute_info.create_state_func =
        [backend_manager](ComputeContext* context, FunctionState* state) {
          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState();
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          p->backend_manager = backend_manager;
          *state = static_cast<FunctionState>(p);
          return 0;
        };
    compute_info.compute_func = [](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
      auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
      try {
        function_state->backend_manager->Compute(context);
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
    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

#ifdef USE_DEVICE_MEMORY
std::vector<AllocatorPtr> OpenVINOExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo npu_allocator_info {
    [this](OrtDevice::DeviceId device_id) {
        return std::make_unique<OVRTAllocator>(global_context_->ie_core.Get(), OrtDevice::NPU, device_id, OpenVINO_RT_NPU);
    },
    0,
  };

  // fill in allocator
  return std::vector<AllocatorPtr>{CreateAllocator(npu_allocator_info)};
}
#endif

}  // namespace onnxruntime
