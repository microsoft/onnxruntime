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
                           SessionContext& session_context,
                           const SubGraphContext& subgraph_context,
                           SharedContext& shared_context,
                           ptr_stream_t& model_stream)
    : session_context_{session_context}, subgraph_context_{subgraph_context}, shared_context_{shared_context} {
  std::string& hw_target = session_context_.device_type;

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

  auto npuw_status =
      std::any_of(device_config.begin(), device_config.end(), [&](const std::pair<std::string, ov::Any>& pair) {
        return (pair.first.find("NPU_USE_NPUW") != std::string::npos) && (pair.second.is<std::string>()) &&
               (pair.second.as<std::string>() == "YES");
      });

  if (npuw_status) {
    LOGS_DEFAULT(INFO) << log_tag << "NPUW Enabled during compilation";
  }

  try {
    // IO_BUFFER is enabled on GPU HW.
    // Pre-requisite is provider_option "context" must be set
#if defined(IO_BUFFER_ENABLED)
    cl_context ctx = static_cast<cl_context>(session_context_.context);
    remote_context_ = new ov::intel_gpu::ocl::ClContext(OVCore::Get()->core, ctx);
    if (subgraph_context_.is_ep_ctx_graph) {
      exe_network_ = OVCore::Get()->ImportModel(*model_stream,
                                                remote_context_,
                                                subgraph_context_.subgraph_name);
      model_stream.reset();  // Delete stream after it is no longer needed
    } else {
      std::shared_ptr<const OVNetwork> ov_model;
      {
        const std::string model = model_proto->SerializeAsString();
        if (!subgraph_context.has_dynamic_input_shape) {
          delete model_proto.release();
        }
        ov_model = CreateOVModel(model, session_context_, const_outputs_map_);
      }
      LOGS_DEFAULT(INFO) << log_tag << "IO Buffering Enabled";
      exe_network_ = OVCore::Get()->CompileModel(
          ov_model, remote_context_, subgraph_context_.subgraph_name);
    }
#else  // !IO_BUFFER_ENABLED
    auto auto_unified_compile = ((hw_target.find("AUTO") == std::string::npos) ||
                                 (session_context_.OpenVINO_Version.at(0) >= 2024 &&
                                  session_context_.OpenVINO_Version.at(1) > 2));
    if (subgraph_context_.is_ep_ctx_graph) {
      // If the blob is held in an EPContext node, then skip FE+Compile
      // and directly move on to creating a backend with the executable blob
      exe_network_ = OVCore::Get()->ImportModel(*model_stream,
                                                hw_target,
                                                device_config,
                                                subgraph_context_.subgraph_name);
      model_stream.reset();  // Delete stream after it is no longer needed
    } else if (!session_context_.has_external_weights &&
               !subgraph_context_.has_dynamic_input_shape &&
               !session_context_.so_context_enable &&
               auto_unified_compile) {
      // Unified OV compile_model is efficient when ov model caching is enabled
      // Unified OV compile_model API is supported with AUTO from version 2024.3 and above
      // Inputs with static dimenstions
      // Not enabled for models with external weights and when ep context is set.
      const std::string model = model_proto->SerializeAsString();
      exe_network_ = OVCore::Get()->CompileModel(model,
                                                 hw_target,
                                                 device_config,
                                                 subgraph_context_.subgraph_name);
    } else {  // For all other types use ov::ov_core read_model() to generate OV IR
              // followed by ov::ov_core compile_model()
      std::shared_ptr<const OVNetwork> ov_model;
      {
        const std::string model = model_proto->SerializeAsString();
        if (!subgraph_context.has_dynamic_input_shape) {
          delete model_proto.release();
        }
        ov_model = CreateOVModel(std::move(model), session_context_, const_outputs_map_);
      }
      exe_network_ = OVCore::Get()->CompileModel(
          ov_model, hw_target, device_config, subgraph_context_.subgraph_name);
    }
#endif
    LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
  } catch (const char* msg) {
    ORT_THROW(msg);
  }

  int num_infer_req = (session_context_.num_of_threads > 0) ? session_context_.num_of_threads : 1;
  std::function<void(OVInferRequestPtr)> initializer = [](OVInferRequestPtr) {};
  auto metadata = shared_context_.shared_weights.metadata;
  if (session_context_.so_share_ep_contexts) {
    initializer = [&metadata](OVInferRequestPtr ir_ptr) {
      const auto input_count = ir_ptr->GetNumInputs();
      for (auto i = 0u; i < input_count; i++) {
        using Key = SharedContext::SharedWeights::Metadata::Key;
        const auto tensor_key = Key{ir_ptr->GetInputTensorName(i)};
        if (metadata.contains(tensor_key)) {
          auto& value = metadata.at(tensor_key);
          ir_ptr->SetTensor(tensor_key.name, value.tensor);
        }
      }
    };
  }
  inferRequestsQueue_ = std::unique_ptr<InferRequestsQueue>(new InferRequestsQueue(exe_network_, num_infer_req, std::move(initializer)));
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
  if (session_context_.precision.find("FP16") != std::string::npos &&
      session_context_.device_type == "GPU") {
    device_config.emplace(ov::hint::inference_precision("f16"));
  }
  if (session_context_.precision.find("FP32") != std::string::npos) {
    device_config.emplace(ov::hint::inference_precision("f32"));
  }
  if (session_context_.precision.find("ACCURACY") != std::string::npos &&
      session_context_.device_type.find("GPU") != std::string::npos) {
    if (session_context_.OpenVINO_Version.at(0) >= 2024) {
      device_config.emplace(ov::hint::inference_precision(ov::element::undefined));
      device_config.emplace(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    } else {
      if (!subgraph_context_.model_precision.empty())
        device_config.emplace(ov::hint::inference_precision(subgraph_context_.model_precision));
    }
  }
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    device_config.emplace(ov::enable_profiling(true));
  }
#endif

  // Set a priority level for the current workload for preemption;  default priority is "DEFAULT"
  // CPU Plugin doesn't support workload priority
  if (session_context_.device_type.find("CPU") == std::string::npos)
    device_config.emplace(ov::hint::model_priority(session_context_.model_priority));

  if (session_context_.device_type.find("NPU") != std::string::npos) {
    std::pair<std::string, ov::Any> device_property;
    device_property = std::make_pair("NPU_COMPILER_TYPE", "DRIVER");

    const std::string env_npu_compiler_type = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_NPU_COMPILER_TYPE");
    if (!env_npu_compiler_type.empty()) {
      device_property = std::make_pair("NPU_COMPILER_TYPE", env_npu_compiler_type);
    }
    device_config.emplace(ov::device::properties("NPU", device_property));
#if (((OPENVINO_VERSION_MAJOR == 2024) && (OPENVINO_VERSION_MINOR > 3)) || (OPENVINO_VERSION_MAJOR > 2024))
    if (session_context_.so_context_enable) {
      OVCore::Get()->core.set_property("NPU", ov::intel_npu::bypass_umd_caching(true));
    }
#endif
  }

  if (!session_context_.load_config.empty()) {
    const std::map<std::string, ov::AnyMap>& target_config = session_context_.load_config;

    if (session_context_.device_type.find("NPU") != std::string::npos) {
      auto npuw_config = target_config.at("NPU");

      // Check if "NPU_USE_NPUW" exists and is set to "YES"
      auto npu_use_npuw_it = npuw_config.find("NPU_USE_NPUW");
      if (npu_use_npuw_it != npuw_config.end() &&
          npu_use_npuw_it->second.is<std::string>() &&
          npu_use_npuw_it->second.as<std::string>() == "YES") {
        // Only add NPUW-related keys if NPU_USE_NPUW is "YES"
        for (const auto& [key, value] : npuw_config) {
          if (key.find("NPUW") != std::string::npos) {
            if (!value.is<std::string>()) {
              LOGS_DEFAULT(ERROR) << "Invalid value type for key: " << key;
              continue;
            }
            device_config[key] = value;
          }
        }
      } else {
        // Check if there are any "NPUW" keys and log a warning
        if (std::any_of(npuw_config.begin(), npuw_config.end(),
                        [&](const auto& pair) { return pair.first.find("NPUW") != std::string::npos; })) {
          LOGS_DEFAULT(WARNING) << "Skipping NPUW-related configurations as NPU_USE_NPUW is not set to 'YES'.";
        }
      }
    }

    // Parse device types like "AUTO:CPU,GPU" and extract individual devices
    auto parse_individual_devices = [&](const std::string& device_type) -> std::vector<std::string> {
      std::vector<std::string> devices;
      auto delimiter_pos = device_type.find(':');
      if (delimiter_pos != std::string::npos) {
        std::stringstream str_stream(device_type.substr(delimiter_pos + 1));
        std::string device;
        while (std::getline(str_stream, device, ',')) {
          devices.emplace_back(device);
        }
      } else {
        devices.emplace_back(device_type);
      }
      return devices;
    };

    // Check if a property is supported and mutable
    auto is_supported_and_mutable = [&](const std::string& key,
                                        const std::vector<ov::PropertyName>& supported_config) -> bool {
      auto it = std::find_if(supported_config.begin(), supported_config.end(), [&](const ov::PropertyName& property) {
        return property == key && property.is_mutable();
      });
      return it != supported_config.end();
    };

    // Set properties if they are valid, else log a warning if the property is missing or immutable by skipping the same
    auto set_target_properties = [&](const std::string& device, const ov::AnyMap& config_options,
                                     const std::vector<ov::PropertyName>& supported_properties) {
      for (const auto& [key, value] : config_options) {
        if (key.find("NPUW") != std::string::npos) {
          continue;
        }
        if (is_supported_and_mutable(key, supported_properties)) {
          OVCore::Get()->core.set_property(device, ov::AnyMap{{key, value}});
        } else {
          LOGS_DEFAULT(WARNING) << "WARNING: Property \"" << key
                                << "\" is either unsupported in current OpenVINO version"
                                << " or property is immutable for target device \""
                                << device << "\". Skipping setting this property.";
        }
      }
    };

    // Check if the device type is AUTO, HETERO, or MULTI
    if (session_context_.device_type.find("AUTO") == 0 ||
        session_context_.device_type.find("HETERO") == 0 ||
        session_context_.device_type.find("MULTI") == 0) {
      // Parse individual devices (e.g., "AUTO:CPU,GPU" -> ["CPU", "GPU"])
      auto individual_devices = parse_individual_devices(session_context_.device_type);
      // Set properties only for individual devices (e.g., "CPU", "GPU")
      for (const std::string& device : individual_devices) {
        if (target_config.count(device)) {
          // Get supported properties for each individual device
          auto device_properties = OVCore::Get()->core.get_property(device, ov::supported_properties);
          // Set properties for the device
          set_target_properties(device, target_config.at(device), device_properties);
        }
      }
    } else {
      if (target_config.count(session_context_.device_type)) {
        auto supported_properties = OVCore::Get()->core.get_property(session_context_.device_type,
                                                                     ov::supported_properties);
        set_target_properties(session_context_.device_type,
                              target_config.at(session_context_.device_type), supported_properties);
      }
    }
  }
}

void BasicBackend::EnableCaching() {
  // cache_dir argument has no effect when working with an embed-mode EPContext Graph
  if (subgraph_context_.is_ep_ctx_graph) return;

  if (!session_context_.cache_dir.empty() && !session_context_.so_context_enable) {
    LOGS_DEFAULT(INFO) << log_tag << "Enables Caching";
    OVCore::Get()->SetCache(session_context_.cache_dir.string());
  }
}

void BasicBackend::EnableGPUThrottling(ov::AnyMap& device_config) {
  if (session_context_.enable_opencl_throttling == true &&
      session_context_.device_type.find("GPU") != std::string::npos) {
    LOGS_DEFAULT(INFO) << log_tag << "Enabled OpenCL queue throttling for GPU device";
    std::pair<std::string, ov::Any> device_property;
    device_property = std::make_pair("PLUGIN_THROTTLE", "1");
    device_config.emplace(ov::device::properties("GPU_CONFIG_KEY", device_property));
  }
}

void BasicBackend::EnableStreams() {
  // Return silently for NPU as it's currently treated as a read-only flag by the NPU plugin
  // and throws an exception for the same
  if (session_context_.device_type.find("NPU") != std::string::npos)
    return;

  // Streams can be set only if the device is not one of AUTO, MULTI, or HETERO
  // Throw an exception if the user tries to set num_streams for these devices
  if ((session_context_.device_type.find("MULTI") != std::string::npos) ||
      (session_context_.device_type.find("HETERO") != std::string::npos) ||
      (session_context_.device_type.find("AUTO") != std::string::npos)) {
    if (session_context_.num_streams != 1) {
      ORT_THROW(log_tag + "Cannot set NUM_STREAMS to " +
                std::to_string(session_context_.num_streams) + " for device " + session_context_.device_type);
    }
    // Do nothing
  } else {
    OVCore::Get()->SetStreams(session_context_.device_type, session_context_.num_streams);
  }
}

void BasicBackend::SetNumThreads(ov::AnyMap& device_config) {
  // inference_num_threads is applicable only for the CPU device
  if (session_context_.device_type.find("CPU") != std::string::npos)
    device_config.emplace(ov::inference_num_threads(session_context_.num_of_threads));
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void BasicBackend::StartAsyncInference(Ort::KernelContext& context, OVInferRequestPtr infer_request) {
  try {
    auto ov_input_info = exe_network_.Get().inputs();

    // Loop over subgraph original input names to find the correspondent OV input name
    for (const auto& [onnx_input_name, onnx_input_index] : subgraph_context_.input_names) {
      std::string input_name{};
      uint32_t input_idx = 0;
      for (uint32_t index = 0; const auto& ov_input : ov_input_info) {
        if (ov_input.get_names().contains(onnx_input_name)) {
          input_name = onnx_input_name;
          input_idx = index;
          break;
        }
        index++;
      }
      ORT_ENFORCE(!input_name.empty(), log_tag,
                  "Input names mismatch between OpenVINO and ONNX. ", onnx_input_name,
                  " doesn't exist in the list of OpenVINO input tensor names");
      size_t batch_slice_idx = 0;
      if (subgraph_context_.has_dynamic_input_shape &&
          !session_context_.disable_dynamic_shapes &&
          (session_context_.device_type.find("CPU") != std::string::npos ||
           session_context_.device_type.find("GPU") != std::string::npos)) {
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
        const auto& input = ov_input_info.at(input_idx);
        OVTensorPtr tensor_ptr;
        // avoid input copies on the CPU device
        if (session_context_.device_type.find("CPU") != std::string::npos) {
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
        if ((session_context_.device_type.find("CPU") != std::string::npos ||
             session_context_.device_type.find("GPU") != std::string::npos)) {
          OVTensorPtr graph_input_blob;
          try {
            graph_input_blob = infer_request->GetTensor(input_name);
          } catch (const char* msg) {
            ORT_THROW(msg);
          }
          FillInputBlob(std::move(graph_input_blob), batch_slice_idx, std::move(input_name), context, subgraph_context_);
        } else {
          auto tensor = context.GetInput(subgraph_context_.input_names.at(input_name));
          ort_tensor_key_t ort_tensor_key{input_name};
          auto it = ort_ov_tensor_map.find(ort_tensor_key);
          if ((it == ort_ov_tensor_map.end()) ||
              (it != ort_ov_tensor_map.end() && (it->second.ort_ptr != tensor.GetTensorRawData()))) {
            ov_tensor_data_t ov_tensor_data;
            const auto& input = ov_input_info.at(input_idx);
            ov_tensor_data.tensor_ptr = std::make_shared<ov::Tensor>(input.get_element_type(), input.get_shape(),
                                                                     const_cast<void*>(tensor.GetTensorRawData()));

            ov_tensor_data.ort_ptr = tensor.GetTensorRawData();
            ort_ov_tensor_map[ort_tensor_key] = ov_tensor_data;

            try {
              infer_request->SetTensor(std::move(input_name), ov_tensor_data.tensor_ptr);
            } catch (const char* msg) {
              ORT_THROW(msg);
            }
          }
        }
      }
    }  // Loop subgraph original input names

    if (session_context_.device_type.find("NPU") != std::string::npos) {
      // Set the output blob as remote blob
      auto graph_output_info = exe_network_.Get().outputs();
      auto output_idx = 0;
      for (auto output_info_iter = graph_output_info.begin();
           output_info_iter != graph_output_info.end(); ++output_info_iter) {
        auto output_names = output_info_iter->get_names();
        std::string onnx_output_name;
        std::string output_name;
        // using the output name retrieved from ONNX original to match with the output names returned by OV tensors
        for (auto it = subgraph_context_.output_names.begin(); it != subgraph_context_.output_names.end(); ++it) {
          onnx_output_name = it->first;
          if (output_names.find(onnx_output_name) != output_names.end()) {
            // Assigning the output_name
            output_name = it->first;
            break;
          }
        }
        size_t batch_size = 1;
        Ort::UnownedValue tensor = GetOutputTensor(context,
                                                   batch_size,
                                                   infer_request,
                                                   output_name,
                                                   subgraph_context_.output_names);
        ort_tensor_key_t ort_tensor_key{output_name};
        const auto& it = ort_ov_tensor_map.find(ort_tensor_key);
        if ((it == ort_ov_tensor_map.end()) ||
            (it != ort_ov_tensor_map.end() && (it->second.ort_ptr != tensor.GetTensorRawData()))) {
          ov_tensor_data_t ov_tensor_data;
          const auto& output = graph_output_info.at(output_idx);
          ov_tensor_data.ort_ptr = tensor.GetTensorRawData();
          ov_tensor_data.tensor_ptr = std::make_shared<ov::Tensor>(output.get_element_type(), output.get_shape(),
                                                                   const_cast<void*>(tensor.GetTensorRawData()));
          ort_ov_tensor_map[ort_tensor_key] = ov_tensor_data;

          try {
            infer_request->SetTensor(std::move(output_name), ov_tensor_data.tensor_ptr);
          } catch (const char* msg) {
            ORT_THROW(msg);
          }
        }
        output_idx++;
      }
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
        ORT_THROW(
            log_tag +
            "Output names mismatch between OpenVINO and ONNX. "
            "[ONNX Output: ] " +
            onnx_output_name +
            " doesn't exist in the "
            "list of OpenVINO output tensor names");
      }
      if ((session_context_.device_type.find("CPU") != std::string::npos ||
           session_context_.device_type.find("GPU") != std::string::npos)) {
        try {
          graph_output_blob = infer_request->GetTensor(output_name);
        } catch (const char* msg) {
          ORT_THROW(msg);
        }
        size_t batch_size = 1;
        Ort::UnownedValue output_tensor =
            GetOutputTensor(context, batch_size, infer_request, std::move(output_name), subgraph_context_.output_names);
        auto mem_info = output_tensor.GetTensorMemoryInfo();
        if (mem_info.GetAllocatorName() == OpenVINO_GPU) {
          return;
        } else {
          size_t batch_slice = 0;
          FillOutputBlob(std::move(graph_output_blob), output_tensor, batch_slice);
        }
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
    if ((session_context_.device_type.find("GPU") != std::string::npos) &&
        (session_context_.context != nullptr) && session_context_.is_wholly_supported_graph) {
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
      std::string& hw_target = session_context_.device_type;
      printPerformanceCounts(std::move(infer_request_), std::cout, hw_target);
    }
#endif
#endif
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
