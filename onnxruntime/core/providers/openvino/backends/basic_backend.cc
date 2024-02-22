// Copyright (C) 2019-2022 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "../backend_utils.h"
#include "basic_backend.h"
#include "../backend_manager.h"

namespace onnxruntime {

namespace openvino_ep {

using namespace backend_utils;

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                           GlobalContext& global_context,
                           const SubGraphContext& subgraph_context)
    : global_context_(global_context), subgraph_context_(subgraph_context) {
  std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
  if (ValidateSubgraph(const_outputs_map_))
    return;

  // OV Config
  ov::AnyMap device_config;
  PopulateConfigValue(device_config);

  // Enable caching
  EnableCaching();

  // Setting OpenCL queue throttling for GPU
  EnableGPUThrottling(device_config);

  // Enable streams; default=1 unless ovverriden by user config
  EnableStreams();

  // Set the inference_num_threads property of the CPU
  SetNumThreads(device_config);

#ifndef NDEBUG
  if (IsDebugEnabled()) {
    std::string file_name = subgraph_context.subgraph_name + "_static.onnx";
    std::fstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto.SerializeToOstream(outfile);
  }
#endif
  try {
    std::string dev_prec = global_context.device_type + "_" + global_context_.precision_str;
    if (global_context.is_wholly_supported_graph) {
#if defined(IO_BUFFER_ENABLED)
      if ((global_context.device_type.find("GPU") != std::string::npos) &&
          (global_context_.context != nullptr)) {
        LOGS_DEFAULT(INFO) << log_tag << "IO Buffering Enabled";
        cl_context ctx = static_cast<cl_context>(global_context_.context);
        remote_context_ = new ov::intel_gpu::ocl::ClContext(global_context_.ie_core.Get(), ctx);
        ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
        exe_network_ = global_context_.ie_core.LoadNetwork(
            ie_cnn_network_, remote_context_, subgraph_context_.subgraph_name);
        LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
      } else {
        ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
        exe_network_ = global_context_.ie_core.LoadNetwork(
            ie_cnn_network_, hw_target, device_config, subgraph_context_.subgraph_name);
        LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
      }
#else
      if (!subgraph_context_.has_dynamic_input_shape &&
          global_context_.onnx_model_path_name != "" &&
          dev_prec != "CPU_FP16") {
        exe_network_ = global_context_.ie_core.LoadNetwork(global_context_.onnx_model_path_name,
                                                           hw_target,
                                                           device_config,
                                                           subgraph_context_.subgraph_name);
        LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
      } else {
        ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
        exe_network_ = global_context_.ie_core.LoadNetwork(
            ie_cnn_network_, hw_target, device_config, subgraph_context_.subgraph_name);
        LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
      }
#endif
    } else {
      ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
      exe_network_ = global_context_.ie_core.LoadNetwork(
          ie_cnn_network_, hw_target, device_config, subgraph_context_.subgraph_name);
      LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
    }
  } catch (const char* msg) {
    throw(msg);
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
  if (global_context_.precision_str.find("FP16") != std::string::npos && global_context_.device_type == "GPU") {
    device_config.emplace(ov::hint::inference_precision("f16"));
  }
  if (global_context_.precision_str.find("FP32") != std::string::npos) {
    device_config.emplace(ov::hint::inference_precision("f32"));
  }
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    device_config.emplace(ov::enable_profiling(true));
  }
#endif
  if (global_context_.device_type.find("NPU") != std::string::npos) {
    std::pair<std::string, ov::Any> device_property;
    device_property = std::make_pair("NPU_COMPILER_TYPE", "DRIVER");
    device_config.emplace(ov::device::properties("NPU", device_property));
  }
}

void BasicBackend::EnableCaching() {
  if (!global_context_.cache_dir.empty()) {
    if (global_context_.is_wholly_supported_graph) {
#if defined(OPENVINO_2022_3)
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
      _putenv_s("OV_GPU_CACHE_MODEL", "1");
#else
      setenv("OV_GPU_CACHE_MODEL", "1", 1);
#endif
#endif
    }
    LOGS_DEFAULT(INFO) << log_tag << "Enables Caching";
    global_context_.ie_core.SetCache(global_context_.cache_dir);
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
  // Streams can be set only if the device is not one of AUTO, MULTI, or HETERO
  // Throw an exception if the user tries to set num_streams for these devices
  if ((global_context_.device_type.find("MULTI") != std::string::npos) ||
      (global_context_.device_type.find("HETERO") != std::string::npos) ||
      (global_context_.device_type.find("AUTO") != std::string::npos)) {
    if (global_context_.num_streams != 1) {
      throw(log_tag + "Cannot set NUM_STREAMS to " + std::to_string(global_context_.num_streams) + " for device " + global_context_.device_type);
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
        input_name = onnx_input_name;
      } else {
        throw(log_tag +
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
        auto input = ie_cnn_network_->get_parameters().at(input_idx);
        OVTensorPtr tensor_ptr;
        // avoid input copies on the CPU device
        if (global_context_.device_type.find("CPU") != std::string::npos) {
          tensor_ptr = std::make_shared<ov::Tensor>(input->get_element_type(), input_tensor_shape,
                                                    (void*)tensor_data);
        } else {
          tensor_ptr = std::make_shared<ov::Tensor>(input->get_element_type(), input_tensor_shape);
          FillInputBlob(tensor_ptr, batch_slice_idx, input_name, context, subgraph_context_);
        }

        try {
          infer_request->SetTensor(input_name, tensor_ptr);
        } catch (const char* msg) {
          throw(msg);
        }
      } else {
        OVTensorPtr graph_input_blob;
        try {
          graph_input_blob = infer_request->GetTensor(input_name);
        } catch (const char* msg) {
          throw(msg);
        }
        FillInputBlob(graph_input_blob, batch_slice_idx, input_name, context, subgraph_context_);
      }
      input_idx++;
    }
    // Start Async inference
    infer_request->StartAsync();
  } catch (const char* msg) {
    throw(msg);
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
        throw(log_tag +
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
        auto input = ie_cnn_network_->get_parameters().at(0);
        auto remote_blob = remote_context_->create_tensor(
            input->get_element_type(), input->get_shape(), *shared_buffer_const);
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
        throw std::string(
            log_tag +
            "Output names mismatch between OpenVINO and ONNX. [ONNX Output: ] " +
            onnx_output_name + " doesn't exist in the list of OpenVINO output tensor names");
      }

      size_t batch_size = 1;
      auto tensor = GetOutputTensor(context, batch_size, infer_request, output_name, subgraph_context_.output_names);
      auto mem_info = tensor.GetTensorMemoryInfo();
      // Check if ORT Value wraps a device pointer
      if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
        const void* tensor_data = tensor.GetTensorRawData();
        const cl::Buffer* shared_buffer_const = static_cast<const cl::Buffer*>(tensor_data);
        // Create a shared Blob, set the Infer Request Output Blob
        auto output = ie_cnn_network_->get_results().at(0);
        auto remote_tensor =
            remote_context_->create_tensor(output->get_element_type(), output->get_shape(), *shared_buffer_const);
        ov::Tensor tensor_t = static_cast<ov::Tensor>(remote_tensor);
        OVTensorPtr tensor_ptr = std::make_shared<ov::Tensor>(tensor_t);
        try {
          infer_request->SetTensor(output_name, tensor_ptr);
        } catch (const char* msg) {
          throw(msg);
        }
      }
    }

    // Start Async inference
    infer_request->StartAsync();
  } catch (const char* msg) {
    throw(msg);
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
      OVTensorPtr graph_output_blob;
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
        throw(log_tag +
              "Output names mismatch between OpenVINO and ONNX. "
              "[ONNX Output: ] " +
              onnx_output_name +
              " doesn't exist in the "
              "list of OpenVINO output tensor names");
      }
      try {
        graph_output_blob = infer_request->GetTensor(output_name);
      } catch (const char* msg) {
        throw(msg);
      }
      size_t batch_size = 1;
      auto output_tensor =
          GetOutputTensor(context, batch_size, infer_request, output_name, subgraph_context_.output_names);
      auto mem_info = output_tensor.GetTensorMemoryInfo();
      if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
        return;
      } else {
        size_t batch_slice = 0;
        FillOutputBlob(graph_output_blob, output_tensor, batch_slice);
      }
    }

    if (!const_outputs_map_.empty()) {
      for (auto item : const_outputs_map_) {
        auto out_name = item.first;
        auto node = item.second;
        auto output_tensor = GetOutputTensor(context, out_name, subgraph_context_.output_names, node);
        auto mem_info = output_tensor.GetTensorMemoryInfo();
        if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
          throw(log_tag + "IO Buffering is not supported for constant subgraphs");
        } else {
          FillOutputsWithConstantData(node, output_tensor);
        }
      }
    }
  } catch (const char* msg) {
    throw(msg);
  }
}

void BasicBackend::Infer(OrtKernelContext* ctx) {
  // Preliminary Thread safety mechanism
  // currently allows a maximum of 8 Infer request's to parallel execute at the same time
  Ort::KernelContext context(ctx);

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";

  if (subgraph_context_.is_constant) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      try {
        auto output_tensor = GetOutputTensor(context, out_name, subgraph_context_.output_names, node);
        FillOutputsWithConstantData(node, output_tensor);
      } catch (std::string const& msg) {
        throw msg;
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
        throw msg;
      }
    } else {
      try {
        StartAsyncInference(context, infer_request);
      } catch (std::string const& msg) {
        throw msg;
      }
    }
#else
    try {
      StartAsyncInference(context, infer_request);
    } catch (std::string const& msg) {
      throw msg;
    }
#endif
    try {
      CompleteAsyncInference(context, infer_request);
    } catch (std::string const& msg) {
      throw msg;
    }

    // Get Output tensors
    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
    // Enable CI Logs
    if (IsCILogEnabled()) {
      std::cout << "Inference successful" << std::endl;
    }

    // Once the inference is completed, the infer_request becomes free and is placed back into pool of infer_requests_
    inferRequestsQueue_->putIdleRequest(infer_request);
#ifndef NDEBUG
#ifndef IO_BUFFER_ENABLED  // Printing performance counts is disabled when IO_BUFFER_ENABLED
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      inferRequestsQueue_->printstatus();  // Printing the elements of infer_requests_ vector pool only in debug mode
      std::string& hw_target =
          (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
      printPerformanceCounts(infer_request, std::cout, hw_target);
    }
#endif
#endif
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
