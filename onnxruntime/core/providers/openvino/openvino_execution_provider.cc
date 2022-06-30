// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "openvino_execution_provider.h"
#include "contexts.h"
#include "backend_manager.h"
#include "ov_versions/capabilities.h"

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const OpenVINOExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider} {
  openvino_ep::BackendManager::GetGlobalContext().device_type = info.device_type_;
  openvino_ep::BackendManager::GetGlobalContext().precision_str = info.precision_;
  openvino_ep::BackendManager::GetGlobalContext().enable_vpu_fast_compile = info.enable_vpu_fast_compile_;
  openvino_ep::BackendManager::GetGlobalContext().use_compiled_network = info.use_compiled_network_;
  openvino_ep::BackendManager::GetGlobalContext().blob_dump_path = info.blob_dump_path_;
  openvino_ep::BackendManager::GetGlobalContext().context = info.context_;
  openvino_ep::BackendManager::GetGlobalContext().enable_opencl_throttling = info.enable_opencl_throttling_;
  openvino_ep::BackendManager::GetGlobalContext().enable_dynamic_shapes = info.enable_dynamic_shapes_;


  if ((int)info.num_of_threads_ <= 0) {
    openvino_ep::BackendManager::GetGlobalContext().num_of_threads = 8;
  } else {
    openvino_ep::BackendManager::GetGlobalContext().num_of_threads = info.num_of_threads_;
  }
  //to check if target device is available
  //using ie_core capability GetAvailableDevices to fetch list of devices plugged in
  bool device_found = false;
  bool device_id_found = false;
  auto available_devices = openvino_ep::BackendManager::GetGlobalContext().ie_core.GetAvailableDevices();
  //Checking for device_type configuration
  if (info.device_type_ != "") {
    if(info.device_type_== "CPU" || info.device_type_== "GPU" || info.device_type_== "MYRIAD") {
      for (auto device : available_devices){
        if (device.rfind(info.device_type_, 0) == 0){
          if(info.device_type_== "GPU" && (info.precision_ == "FP32" || info.precision_ == "FP16")) {
              device_found = true;
              break;
          }
          if(info.device_type_== "CPU" && info.precision_ == "FP32") {
            device_found = true;
            break;
          }
          if(info.device_type_== "MYRIAD" && info.precision_ == "FP16") {
            device_found = true;
            break;
          }
        }
      }
    } else {
      device_found = true;
    }
  }
  if (!device_found) {
    std::string err_msg = std::string("Device Type not found : ") + info.device_type_ + "\nChoose the right precision with one of:\n";
    for (auto device : available_devices) {
      err_msg = err_msg + device + "\n";
    }
    ORT_THROW(err_msg);
  }
  //Checking for device_id configuration
  if (info.device_id_ != "") {
    for (auto device : available_devices) {
      if (device.rfind(info.device_id_, 0) == 0){
        if(info.device_id_== "MYRIAD" || info.device_id_== "CPU" || info.device_id_== "GPU"){
          LOGS_DEFAULT(INFO) << "[OpenVINO-EP]"<< "Switching to Device ID: " << info.device_id_;
          device_id_found = true;
          break;
        }
      }
    }
    if (!device_id_found) {
      std::string err_msg = std::string("Device ID not found : ") + info.device_id_ + "\nChoose one of:\n";
      for (auto device : available_devices) {
        err_msg = err_msg + device + "\n";
      }
      ORT_THROW(err_msg);
    }
  }
  openvino_ep::BackendManager::GetGlobalContext().device_id = info.device_id_;

  AllocatorCreationInfo device_info(
      [](int) {
        return CreateCPUAllocator(OrtMemoryInfo(OpenVINO_CPU, OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>>
OpenVINOExecutionProvider::GetCapability(const GraphViewer& graph_viewer, const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);

  std::vector<std::unique_ptr<ComputeCapability>> result;
  //Enable CI Logs
  if (!(GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG").empty())) {
    std::cout << "In the OpenVINO EP" << std::endl;
  }
  openvino_ep::BackendManager::GetGlobalContext().onnx_model_name = graph_viewer.Name();
#ifdef _WIN32
  std::wstring onnx_path = graph_viewer.ModelPath().ToPathString();
  openvino_ep::BackendManager::GetGlobalContext().onnx_model_path_name = std::string(onnx_path.begin(), onnx_path.end());
#else
  openvino_ep::BackendManager::GetGlobalContext().onnx_model_path_name = graph_viewer.ModelPath().ToPathString();
#endif
  openvino_ep::BackendManager::GetGlobalContext().onnx_opset_version = graph_viewer.DomainToVersionMap().at(kOnnxDomain);

#if defined (OPENVINO_2021_2)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_2");
  result = obj.Execute();
#elif defined (OPENVINO_2021_3)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_3");
  result = obj.Execute();
#elif defined (OPENVINO_2021_4)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_4");
  result = obj.Execute();
#elif defined (OPENVINO_2022_1)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2022_1");
  result = obj.Execute();
#endif

  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    NodeComputeInfo compute_info;

#if defined(OV_API_20)
    openvino_ep::BackendManager::GetGlobalContext().use_api_2 = true;
#else
    openvino_ep::BackendManager::GetGlobalContext().use_api_2 = false;
#endif 

    std::shared_ptr<openvino_ep::BackendManager> backend_manager = std::make_shared<openvino_ep::BackendManager>(fused_node, graph_body_viewer, *GetLogger());

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
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
      try {
        function_state->backend_manager->Compute(*api, context);
      } catch (const char* msg) {
        return common::Status(common::ONNXRUNTIME, common::FAIL, msg);
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

}  // namespace onnxruntime
