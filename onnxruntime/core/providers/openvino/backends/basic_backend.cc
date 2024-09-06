// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backends/basic_backend.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/backend_manager.h"

namespace onnxruntime {

namespace openvino_ep {

using namespace backend_utils;

BasicBackend::BasicBackend(std::unique_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
                           GlobalContext& global_context,
                           const SubGraphContext& subgraph_context,
                           EPCtxHandler& ep_ctx_handle)
    : global_context_(global_context), subgraph_context_(subgraph_context) {
  std::string& hw_target = global_context_.device_type;

  is_ep_ctx_graph_ = ep_ctx_handle.IsValidOVEPCtxGraph();

  if (ValidateSubgraph(const_outputs_map_))
    return;

  // OV Config
  ov::AnyMap device_config;
  PopulateConfigValue(device_config);

  // Enable caching
  EnableCaching(device_config);

  // Setting OpenCL queue throttling for GPU
  EnableGPUThrottling(device_config);

  // Enable streams; default=1 unless ovverriden by user config
  EnableStreams();

  // Set the inference_num_threads property of the CPU
  SetNumThreads(device_config);

  try {
    std::string dev_prec = global_context.device_type + "_" + global_context_.precision_str;

    if (global_context.is_wholly_supported_graph) {  // Full graph is supported
#if defined(IO_BUFFER_ENABLED)
      if (is_ep_ctx_graph_) {
        std::istringstream model_stream(ep_ctx_handle.GetModelBlobString());
        exe_network_ = global_context_.ie_core.ImportModel(model_stream,
                                                           remote_context_,
                                                           subgraph_context_.subgraph_name);
      } else if ((global_context.device_type.find("GPU") != std::string::npos) &&
                 (global_context_.context != nullptr)) {
        LOGS_DEFAULT(INFO) << log_tag << "IO Buffering Enabled";
        cl_context ctx = static_cast<cl_context>(global_context_.context);
        remote_context_ = new ov::intel_gpu::ocl::ClContext(global_context_.ie_core.Get(), ctx);
        ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
        exe_network_ = global_context_.ie_core.CompileModel(
            ie_cnn_network_, remote_context_, subgraph_context_.subgraph_name);
      } else {
        ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
        exe_network_ = global_context_.ie_core.CompileModel(
            ie_cnn_network_, hw_target, device_config, subgraph_context_.subgraph_name);
      }
#else  // !IO_BUFFER_ENABLED
      std::string prec_str = (global_context_.precision_str != "ACCURACY") ? global_context_.precision_str : global_context_.model_precision;
      if (is_ep_ctx_graph_) {
        // If the blob is held in an EPContext node, then skip FE+Compile
        // and directly move on to creating a backend with the executable blob
        exe_network_ = global_context_.ie_core.ImportModel(ep_ctx_handle.GetModelBlobStream(),
                                                           hw_target,
                                                           device_config,
                                                           global_context_.ep_context_embed_mode,
                                                           subgraph_context_.subgraph_name);
        ie_cnn_network_ = exe_network_.Get().get_runtime_model();
      } else if (global_context_.export_ep_ctx_blob &&
                 hw_target.find("NPU") != std::string::npos) {
        std::shared_ptr<ov::Model> ov_model;
        {
          const std::string model = model_proto->SerializeAsString();
          if (!subgraph_context.has_dynamic_input_shape) {
            delete model_proto.release();
          }
          ov_model = global_context_.ie_core.Get().read_model(model, ov::Tensor());
        }
        exe_network_ = OVExeNetwork(global_context_.ie_core.Get().compile_model(ov_model, hw_target, device_config));
      } else if ((!subgraph_context_.has_dynamic_input_shape) &&
                 ((hw_target.find("AUTO") == std::string::npos) ||
                  (global_context_.OpenVINO_Version.at(0) >= 2024 && global_context_.OpenVINO_Version.at(1) > 2))) {
        // Optimized OV compile_model API is supported with AUTO from version 2024.3 and above
        // Inputs with static dimenstions
        const std::string model = model_proto->SerializeAsString();
        exe_network_ = global_context_.ie_core.CompileModel(model,
                                                            hw_target,
                                                            device_config,
                                                            subgraph_context_.subgraph_name);
      } else {  // For all other types use ov::Model Type
        ie_cnn_network_ = CreateOVModel(*model_proto, global_context_, const_outputs_map_);
        exe_network_ = global_context_.ie_core.CompileModel(
            ie_cnn_network_, hw_target, device_config, subgraph_context_.subgraph_name);
      }
#endif
    } else {  // Full graph is not supported
      ie_cnn_network_ = CreateOVModel(*model_proto, global_context_, const_outputs_map_);
      exe_network_ = global_context_.ie_core.CompileModel(
          ie_cnn_network_, hw_target, device_config, subgraph_context_.subgraph_name);
    }
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
  } catch (const char* msg) {
    ORT_THROW(msg);
  }

  inferRequestsQueue_ = std::unique_ptr<InferRequestsQueue>(new InferRequestsQueue(exe_network_, 1));
}

bool BasicBackend::ValidateSubgraph(std::map<std::string, std::shared_ptr<ov::Node>>& const_outputs_map) {
  if (const_outputs_map.size() == subgraph_context_.output_names.size())
    subgraph_context_.is_constant = true;
  if (subgraph_context_.is_constant) {
    LOGS_DEFAULT(INFO) << log_tag << "The subgraph is a const. Directly moving to Infer stage.";
    return true;
  }
  return false;
}

void BasicBackend::PopulateConfigValue(ov::AnyMap& device_config) {
  device_config = {};
  // Set inference precision based on device precision for OV backend
  if (global_context_.precision_str.find("FP16") != std::string::npos &&
      global_context_.device_type == "GPU") {
    device_config.emplace(ov::hint::inference_precision("f16"));
  }
  if (global_context_.precision_str.find("FP32") != std::string::npos) {
    device_config.emplace(ov::hint::inference_precision("f32"));
  }
  if (global_context_.precision_str.find("ACCURACY") != std::string::npos &&
      global_context_.device_type == "GPU") {
    if (global_context_.OpenVINO_Version.at(0) >= 2024 && global_context_.OpenVINO_Version.at(1) >= 1) {
      device_config.emplace(ov::hint::inference_precision(ov::element::undefined));
      device_config.emplace(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    } else {
      if (global_context_.model_precision != "")
        device_config.emplace(ov::hint::inference_precision(global_context_.model_precision));
    }
  }
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    device_config.emplace(ov::enable_profiling(true));
  }
#endif

  // Set a priority level for the current workload for preemption;  default priority is "DEFAULT"
  // CPU Plugin doesn't support workload priority
  if (global_context_.device_type.find("CPU") == std::string::npos)
    device_config.emplace(ov::hint::model_priority(global_context_.model_priority));

  if (global_context_.device_type.find("NPU") != std::string::npos) {
    std::pair<std::string, ov::Any> device_property;
    device_property = std::make_pair("NPU_COMPILER_TYPE", "DRIVER");

    const std::string env_npu_compiler_type = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_NPU_COMPILER_TYPE");
    if (!env_npu_compiler_type.empty()) {
      device_property = std::make_pair("NPU_COMPILER_TYPE", env_npu_compiler_type);
    }
    device_config.emplace(ov::device::properties("NPU", device_property));
#if (OPENVINO_VERSION_MAJOR >= 2024) && (OPENVINO_VERSION_MINOR > 3)
    if (global_context_.export_ep_ctx_blob) {
      global_context_.ie_core.Get().set_property("NPU", ov::intel_npu::bypass_umd_caching(true));
    }
#endif
  }
}

void BasicBackend::EnableCaching(ov::AnyMap& device_config) {
  // cache_dir argument has no effect when working with an embed-mode EPContext Graph
  if (is_ep_ctx_graph_) return;

  if (!global_context_.cache_dir.empty() && !global_context_.export_ep_ctx_blob) {
    LOGS_DEFAULT(INFO) << log_tag << "Enables Caching";
    if (global_context_.device_type.find("AUTO:GPU") != std::string::npos) {
      std::pair<std::string, ov::Any> device_property;
      device_property = std::make_pair("CACHE_DIR", global_context_.cache_dir);
      device_config.emplace(ov::device::properties("GPU", device_property));
    } else {
      global_context_.ie_core.SetCache(global_context_.cache_dir);
    }
  }
}

void BasicBackend::EnableGPUThrottling(ov::AnyMap& device_config) {
  if (global_context_.enable_opencl_throttling == true &&
      global_context_.device_type.find("GPU") != std::string::npos) {
    LOGS_DEFAULT(INFO) << log_tag << "Enabled OpenCL queue throttling for GPU device";
    std::pair<std::string, ov::Any> device_property;
    device_property = std::make_pair("PLUGIN_THROTTLE", "1");
    device_config.emplace(ov::device::properties("GPU_CONFIG_KEY", device_property));
  }
}

void BasicBackend::EnableStreams() {
  // Return silently for NPU as it's currently treated as a read-only flag by the NPU plugin
  // and throws an exception for the same
  if (global_context_.device_type.find("NPU") != std::string::npos)
    return;

  // Streams can be set only if the device is not one of AUTO, MULTI, or HETERO
  // Throw an exception if the user tries to set num_streams for these devices
  if ((global_context_.device_type.find("MULTI") != std::string::npos) ||
      (global_context_.device_type.find("HETERO") != std::string::npos) ||
      (global_context_.device_type.find("AUTO") != std::string::npos)) {
    if (global_context_.num_streams != 1) {
      ORT_THROW(log_tag + "Cannot set NUM_STREAMS to " +
                std::to_string(global_context_.num_streams) + " for device " + global_context_.device_type);
    }
    // Do nothing
  } else {
    global_context_.ie_core.SetStreams(global_context_.device_type, global_context_.num_streams);
  }
}

void BasicBackend::SetNumThreads(ov::AnyMap& device_config) {
  // inference_num_threads is applicable only for the CPU device
  if (global_context_.device_type.find("CPU") != std::string::npos)
    device_config.emplace(ov::inference_num_threads(global_context_.num_of_threads));
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void BasicBackend::StartAsyncInference(Ort::KernelContext& context, OVInferRequestPtr infer_request) {
  try {
    auto graph_input_info = exe_network_.Get().inputs();
    int input_idx = 0;
    for (auto input_info_iter = graph_input_info.begin();
         input_info_iter != graph_input_info.end(); ++input_info_iter) {
      auto input_names = input_info_iter->get_names();
      std::string onnx_input_name;
      std::string input_name;
      // use names retrieved from original ONNX model to assign the right onnx input name for the graph
      for (auto it = subgraph_context_.input_names.begin(); it != subgraph_context_.input_names.end(); ++it) {
        if (it->second == input_idx) {
          onnx_input_name = it->first;
          break;
        }
      }
      // using the input name retrieved from ONNX original to match with the input names returned by OV tensors
      if (input_names.find(onnx_input_name) != input_names.end()) {
        input_name = std::move(onnx_input_name);
      } else {
        ORT_THROW(log_tag +
                  "Input names mismatch between OpenVINO and ONNX. " + onnx_input_name +
                  " doesn't exist in the list of OpenVINO input tensor names");
      }
      size_t batch_slice_idx = 0;
      if (subgraph_context_.has_dynamic_input_shape &&
          !global_context_.disable_dynamic_shapes &&
          (global_context_.device_type.find("CPU") != std::string::npos ||
           global_context_.device_type.find("GPU") != std::string::npos)) {
        auto tensor = context.GetInput(subgraph_context_.input_names.at(input_name));
        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
        auto tensor_shape = tensor_info.GetShape();
        auto tensor_size = tensor_shape.size();
        const char* tensor_data = tensor.GetTensorData<char>();
        auto tensor_iter = 0;
        ov::Shape input_tensor_shape = ov::Shape(tensor_size, 0);
        for (auto i = tensor_shape.begin(); i != tensor_shape.end(); ++i) {
          input_tensor_shape[tensor_iter] = *i;
          tensor_iter += 1;
        }
        auto input = graph_input_info.at(input_idx);
        OVTensorPtr tensor_ptr;
        // avoid input copies on the CPU device
        if (global_context_.device_type.find("CPU") != std::string::npos) {
          tensor_ptr = std::make_shared<ov::Tensor>(input.get_element_type(), input_tensor_shape,
                                                    (void*)tensor_data);
        } else {
          tensor_ptr = std::make_shared<ov::Tensor>(input.get_element_type(), input_tensor_shape);
          FillInputBlob(tensor_ptr, batch_slice_idx, input_name, context, subgraph_context_);
        }

        try {
          infer_request->SetTensor(std::move(input_name), tensor_ptr);
        } catch (const char* msg) {
          ORT_THROW(msg);
        }
      } else {
        auto tensor = context.GetInput(subgraph_context_.input_names.at(input_name));
        auto allocator_name = tensor.GetTensorMemoryInfo().GetAllocatorName();
        ov_tensor_data_t ov_tensor_key;
        ort_tensor_key_t ort_tensor_key{tensor.GetTensorRawData(), allocator_name};
        if (const auto& it = ort_ov_tensor_map.find(ort_tensor_key); it != ort_ov_tensor_map.end()) {
          ov_tensor_key = it->second;
        } else {
          // Does this make sense for both types of allocators?
          auto input = graph_input_info.at(input_idx);
          ov_tensor_key.tensor_ptr = std::make_shared<ov::Tensor>(input.get_element_type(), input.get_shape(),
                                                                    (void*)tensor.GetTensorRawData());
          if (allocator_name == OpenVINO_RT_NPU) {
            ov_tensor_key.copy_needed = false;
          } else {
            ov_tensor_key.copy_needed = true;
          }
          ort_ov_tensor_map.emplace(ort_tensor_key, ov_tensor_key);

          try {
            infer_request->SetTensor(input_name, ov_tensor_key.tensor_ptr);
          } catch (const char* msg) {
            ORT_THROW(msg);
          }
        }

        if (ov_tensor_key.copy_needed) {
          const char* ort_tensor_data = tensor.GetTensorData<char>();
          size_t tensor_data_size = ov_tensor_key.tensor_ptr->get_byte_size();
          auto ort_batch_memory_offset = ort_tensor_data + tensor_data_size * batch_slice_idx;
          std::memcpy(ov_tensor_key.tensor_ptr->data(), ort_batch_memory_offset, tensor_data_size);
        }
      }
      input_idx++;
    }

    // Set the output blob as remote blob
    auto graph_output_info = exe_network_.Get().outputs();
    auto output_idx = 0;
    for (auto output_info_iter = graph_output_info.begin();
         output_info_iter != graph_output_info.end(); ++output_info_iter) {
      auto output_names = output_info_iter->get_names();
      std::string onnx_output_name;
      std::string output_name;
      bool output_name_found = false;
      // using the output name retrieved from ONNX original to match with the output names returned by OV tensors
      for (auto it = subgraph_context_.output_names.begin(); it != subgraph_context_.output_names.end(); ++it) {
        onnx_output_name = it->first;
        if (output_names.find(onnx_output_name) != output_names.end()) {
          // Assigning the output_name
          output_name = it->first;
          output_name_found = true;
          break;
        }
      }
      size_t batch_size = 1;
      Ort::UnownedValue tensor = GetOutputTensor(context,
                                                 batch_size,
                                                 infer_request,
                                                 output_name,
                                                 subgraph_context_.output_names);
      auto allocator_name = tensor.GetTensorMemoryInfo().GetAllocatorName();

      ov_tensor_data_t ov_tensor_data;
      ort_tensor_key_t ort_tensor_key{tensor.GetTensorRawData(), allocator_name};
      if (const auto& it = ort_ov_tensor_map.find(ort_tensor_key); it != ort_ov_tensor_map.end()) {
        ov_tensor_data = it->second;
      } else {
        auto output = graph_output_info.at(output_idx);
        ov_tensor_data.tensor_ptr = std::make_shared<ov::Tensor>(output.get_element_type(), output.get_shape(),
                                                                 (void*)tensor.GetTensorRawData());
        if(allocator_name == OpenVINO_RT_NPU) {
          ov_tensor_data.copy_needed = false;
        } else {
          ov_tensor_data.copy_needed = true;
        }
        ort_ov_tensor_map.emplace(ort_tensor_key, ov_tensor_data);

        try {
          infer_request->SetTensor(output_name, ov_tensor_data.tensor_ptr);
        } catch (const char* msg) {
          ORT_THROW(msg);
        }
      }
      output_idx++;
    }

    // Start Async inference
    infer_request->StartAsync();
  } catch (const char* msg) {
    ORT_THROW(msg);
  }
}

#ifdef IO_BUFFER_ENABLED
// Wait for Remote Aynchronous inference completion
void BasicBackend::StartRemoteAsyncInference(Ort::KernelContext& context, OVInferRequestPtr infer_request) {
  try {
    auto graph_input_info = exe_network_.Get().inputs();
    int input_idx = 0;
    for (auto input_info_iter = graph_input_info.begin();
         input_info_iter != graph_input_info.end(); ++input_info_iter) {
      auto input_names = input_info_iter->get_names();
      std::string onnx_input_name;
      std::string input_name;
      // use names retrieved from original ONNX model to assign the right onnx input name for the graph
      for (auto it = subgraph_context_.input_names.begin(); it != subgraph_context_.input_names.end(); ++it) {
        if (it->second == input_idx) {
          onnx_input_name = it->first;
          break;
        }
      }
      // using the input name retrieved from ONNX original to match with the input names returned by OV tensors
      if (input_names.find(onnx_input_name) != input_names.end()) {
        input_name = onnx_input_name;
      } else {
        ORT_THROW(log_tag +
                  "Input names mismatch between OpenVINO and ONNX. " +
                  onnx_input_name +
                  " doesn't exist in the list of OpenVINO input tensor names");
      }
      input_idx++;
      // Kernel Context Input Buffer
      const auto tensor = context.GetInput(subgraph_context_.input_names.at(input_name));
      // If the ORTValue wraps a device pointer
      auto mem_info = tensor.GetTensorMemoryInfo();
      if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
        // Get the shared buffer pointer
        const void* tensor_data = tensor.GetTensorRawData();
        const cl::Buffer* shared_buffer_const = static_cast<const cl::Buffer*>(tensor_data);
        // Create an Input Remote Blob
        auto input = graph_input_info.at(0);
        auto remote_blob = remote_context_->create_tensor(
            input.get_element_type(), input.get_shape(), *shared_buffer_const);
        ov::Tensor tensor_remote = static_cast<ov::Tensor>(remote_blob);
        OVTensorPtr tensor_ptr = std::make_shared<ov::Tensor>(tensor_remote);
        infer_request->SetTensor(input_name, tensor_ptr);
      } else {
        OVTensorPtr graph_input_blob;
        graph_input_blob = infer_request->GetTensor(input_name);
        size_t batch_slice_idx = 0;
        FillInputBlob(graph_input_blob, batch_slice_idx, input_name, context, subgraph_context_);
      }
    }

    // Set the output blob as remote blob
    auto graph_output_info = exe_network_.Get().outputs();
    for (auto output_info_iter = graph_output_info.begin();
         output_info_iter != graph_output_info.end(); ++output_info_iter) {
      auto output_names = output_info_iter->get_names();
      std::string onnx_output_name;
      std::string output_name;
      bool output_name_found = false;
      // using the output name retrieved from ONNX original to match with the output names returned by OV tensors
      for (auto it = subgraph_context_.output_names.begin(); it != subgraph_context_.output_names.end(); ++it) {
        onnx_output_name = it->first;
        if (output_names.find(onnx_output_name) != output_names.end()) {
          // Assigning the output_name
          output_name = it->first;
          output_name_found = true;
          break;
        }
      }
      if (!output_name_found) {
        ORT_THROW(
            log_tag +
            "Output names mismatch between OpenVINO and ONNX. [ONNX Output: ] " +
            onnx_output_name + " doesn't exist in the list of OpenVINO output tensor names");
      }

      size_t batch_size = 1;
      Ort::UnownedValue tensor = GetOutputTensor(context,
                                                 batch_size,
                                                 infer_request,
                                                 output_name,
                                                 subgraph_context_.output_names);
      auto mem_info = tensor.GetTensorMemoryInfo();
      // Check if ORT Value wraps a device pointer
      if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
        const void* tensor_data = tensor.GetTensorRawData();
        const cl::Buffer* shared_buffer_const = static_cast<const cl::Buffer*>(tensor_data);
        // Create a shared Blob, set the Infer Request Output Blob
        auto output = graph_output_info.at(0);
        auto remote_tensor =
            remote_context_->create_tensor(output.get_element_type(), output.get_shape(), *shared_buffer_const);
        ov::Tensor tensor_t = static_cast<ov::Tensor>(remote_tensor);
        OVTensorPtr tensor_ptr = std::make_shared<ov::Tensor>(tensor_t);
        try {
          infer_request->SetTensor(output_name, tensor_ptr);
        } catch (const char* msg) {
          ORT_THROW(msg);
        }
      }
    }

    // Start Async inference
    infer_request->StartAsync();
  } catch (const char* msg) {
    ORT_THROW(msg);
  }
}
#endif

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void BasicBackend::CompleteAsyncInference(Ort::KernelContext& context, OVInferRequestPtr infer_request) {
  // Wait for Async inference completion
  try {
    infer_request->WaitRequest();
    auto graph_output_info = exe_network_.Get().outputs();
    for (auto output_info_iter = graph_output_info.begin();
         output_info_iter != graph_output_info.end(); ++output_info_iter) {
      auto output_names = output_info_iter->get_names();
      std::string onnx_output_name;
      std::string output_name;
      bool output_name_found = false;
      // using the output name retrieved from ONNX original to match with the output names returned by OV tensors
      for (auto it = subgraph_context_.output_names.begin(); it != subgraph_context_.output_names.end(); ++it) {
        onnx_output_name = it->first;
        if (output_names.find(onnx_output_name) != output_names.end()) {
          // Assigning the output_name
          output_name = it->first;
          output_name_found = true;
          break;
        }
      }
      if (!output_name_found) {
        ORT_THROW(
            log_tag +
            "Output names mismatch between OpenVINO and ONNX. "
            "[ONNX Output: ] " +
            onnx_output_name +
            " doesn't exist in the "
            "list of OpenVINO output tensor names");
      }

      size_t batch_size = 1;
      Ort::UnownedValue output_tensor =
          GetOutputTensor(context, batch_size, infer_request, std::move(output_name), subgraph_context_.output_names);
      auto allocator_name = output_tensor.GetTensorMemoryInfo().GetAllocatorName();
      ov_tensor_data_t ov_tensor_data;
      ort_tensor_key_t ort_tensor_key{output_tensor.GetTensorRawData(), allocator_name};
      if (const auto& it = ort_ov_tensor_map.find(ort_tensor_key); it != ort_ov_tensor_map.end()) {
        ov_tensor_data = it->second;
      } else {
        ORT_THROW(log_tag + "Expected all outputs to have associated OV::Tensor's");
      }

      if (ov_tensor_data.copy_needed) {
        auto ort_tensor_data = output_tensor.GetTensorMutableData<char>();
        size_t tensor_data_size = ov_tensor_data.tensor_ptr->get_byte_size();
        auto ort_batch_memory_offset = ort_tensor_data /*+ tensor_data_size * batch_size*/;
        std::memcpy(ort_batch_memory_offset, ov_tensor_data.tensor_ptr->data(), tensor_data_size);
      }
    }

    if (!const_outputs_map_.empty()) {
      for (const auto& item : const_outputs_map_) {
        const auto& out_name = item.first;
        auto node = item.second;
        Ort::UnownedValue output_tensor = GetOutputTensor(context,
                                                          std::move(out_name),
                                                          subgraph_context_.output_names,
                                                          node);
        auto mem_info = output_tensor.GetTensorMemoryInfo();
        if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
          ORT_THROW(log_tag + "IO Buffering is not supported for constant subgraphs");
        } else {
          FillOutputsWithConstantData(std::move(node), output_tensor);
        }
      }
    }
  } catch (const char* msg) {
    ORT_THROW(msg);
  }
}

void BasicBackend::Infer(OrtKernelContext* ctx) {
  // Preliminary Thread safety mechanism
  // currently allows a maximum of 8 Infer request's to parallel execute at the same time
  Ort::KernelContext context(ctx);

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";

  if (subgraph_context_.is_constant) {
    for (const auto& item : const_outputs_map_) {
      std::string out_name = item.first;
      std::shared_ptr<ov::Node> node = item.second;
      try {
        Ort::UnownedValue output_tensor = GetOutputTensor(context,
                                                          std::move(out_name),
                                                          subgraph_context_.output_names,
                                                          node);
        FillOutputsWithConstantData(std::move(node), output_tensor);
      } catch (std::string const& msg) {
        ORT_THROW(msg);
      }
    }
    // Get Output tensors
    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
    // Enable CI Logs
    if (IsCILogEnabled()) {
      std::cout << "Inference successful" << std::endl;
    }

  } else {
    // Requesting for an idle infer_request from a pool of infer_requests_
    OVInferRequestPtr infer_request;
    infer_request = inferRequestsQueue_->getIdleRequest();

#ifdef IO_BUFFER_ENABLED
    if ((global_context_.device_type.find("GPU") != std::string::npos) &&
        (global_context_.context != nullptr) && global_context_.is_wholly_supported_graph) {
      try {
        StartRemoteAsyncInference(context, infer_request);
      } catch (std::string const& msg) {
        ORT_THROW(msg);
      }
    } else {
      try {
        StartAsyncInference(context, infer_request);
      } catch (std::string const& msg) {
        ORT_THROW(msg);
      }
    }
#else
    try {
      StartAsyncInference(context, infer_request);
    } catch (const std::runtime_error& e) {
      ORT_THROW(log_tag + " Exception at StartAsyncInference: " + e.what());
    }
#endif
    try {
      CompleteAsyncInference(context, infer_request);
    } catch (const std::runtime_error& e) {
      ORT_THROW(log_tag + " Exception at CompleteAsyncInference: " + e.what());
    }

    // Get Output tensors
    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
    // Enable CI Logs
    if (IsCILogEnabled()) {
      std::cout << "Inference successful" << std::endl;
    }

    // Create a duplicate infer_request_ shared ptr on the stack in the current local scope,
    // as the infer_request gets freed in the next stage the reference count for the infer_request decrements &
    // thus we dont have any dangling ptr leading to seg faults in the debug mode subsequent execution call
    OVInferRequestPtr infer_request_ = infer_request;

    // Once the inference is completed, the infer_request becomes free and is placed back into pool of infer_requests_
    inferRequestsQueue_->putIdleRequest(std::move(infer_request));
#ifndef NDEBUG
#ifndef IO_BUFFER_ENABLED  // Printing performance counts is disabled when IO_BUFFER_ENABLED
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      inferRequestsQueue_->printstatus();  // Printing the elements of infer_requests_ vector pool only in debug mode
      std::string& hw_target = global_context_.device_type;
      printPerformanceCounts(std::move(infer_request_), std::cout, hw_target);
    }
#endif
#endif
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
