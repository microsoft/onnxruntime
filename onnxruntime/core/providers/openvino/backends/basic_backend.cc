// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <map>
#include <unordered_set>

#include <string>
#include <memory>
#include <sstream>
#include <fstream>
#include <utility>
#include <iostream>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backends/basic_backend.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/ov_stateful_patch_utils.h"

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
  bool enable_causallm = session_context_.enable_causallm;

  if (ValidateSubgraph(const_outputs_map_))
    return;

  // OV Config
  ov::AnyMap device_config;
  PopulateConfigValue(device_config);

  // Enable caching
  EnableCaching();

  // Setting OpenCL queue throttling for GPU
  EnableGPUThrottling(device_config);

  // Enable streams; default=1 unless overridden by user configuration
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
    auto auto_unified_compile = ((hw_target.find("AUTO") == std::string::npos) ||
                                 (session_context_.OpenVINO_Version.at(0) >= 2024 &&
                                  session_context_.OpenVINO_Version.at(1) > 2));
    bool disable_cpu_fallback = !(hw_target.find("NPU") != std::string::npos &&
                                  !session_context_.so_disable_cpu_ep_fallback &&
                                  !subgraph_context_.is_ep_ctx_graph);
    if (subgraph_context_.is_ep_ctx_graph) {
      if (subgraph_context_.is_ep_ctx_ovir_encapsulated) {
        // model_file_path will use so_context_file_path if the onnx_model_path_name is not available,
        // especially in case of CreateSessionFormArray() where user must explicitly
        // specify absolute path for so_context_file_path.
        auto model_file_path = [this]() {
          if (!session_context_.onnx_model_path_name.empty() &&
              std::filesystem::exists(session_context_.onnx_model_path_name)) return session_context_.onnx_model_path_name;

          ORT_ENFORCE(!session_context_.so_context_file_path.empty() &&
                          std::filesystem::path(session_context_.so_context_file_path).is_absolute() &&
                          std::filesystem::exists(session_context_.so_context_file_path),
                      log_tag +
                          "Context file path must be non-empty & absolute, when using CreateSessionFormArray() API explicitly."
                          " Please set a valid absolute path for ep.context_file_path in session options.");
          // Return absolute context file path as input to ImportEPCtxOVIREncapsulation() function.
          return session_context_.so_context_file_path;
        };
        // If the EPContext node with OVIR Encapsulation, then create
        // an executable network from EP_CACHE_CONTEXT using read_model() & compile_model()
        exe_network_ = OVCore::Get()->ImportEPCtxOVIREncapsulation(*model_stream,
                                                                   hw_target,
                                                                   device_config,
                                                                   enable_causallm,
                                                                   model_file_path());
      } else {
        // If the blob is held in an EPContext node, then skip FE+Compile
        // and directly move on to creating a backend with the executable blob
        exe_network_ = OVCore::Get()->ImportModel(*model_stream,
                                                  hw_target,
                                                  device_config,
                                                  subgraph_context_.subgraph_name);
      }
      model_stream.reset();  // Delete stream after it is no longer needed
    } else if (!session_context_.has_external_weights &&
               !subgraph_context_.has_dynamic_input_shape &&
               !session_context_.so_context_enable &&
               session_context_.reshape.empty() &&
               !enable_causallm &&
               auto_unified_compile) {
      // Unified OV compile_model is efficient when ov model caching is enabled
      // Unified OV compile_model API is supported with AUTO from version 2024.3 and above
      // Inputs with static dimensions
      // Not enabled for models with external weights and when ep context is set.
      const std::string model = model_proto->SerializeAsString();
      // we have the serialized string, so we can release model proto to lower the peak memory consumption
      if (disable_cpu_fallback) model_proto.reset();
      exe_network_ = OVCore::Get()->CompileModel(model,
                                                 hw_target,
                                                 device_config,
                                                 subgraph_context_.subgraph_name);
    } else {  // For all other types use ov::ov_core read_model() to generate OV IR
              // followed by ov::ov_core compile_model()
      std::string model = model_proto->SerializeAsString();
      // Reset model proto only when cpu fallback is disabled or when the model has dynamic input shapes.
      // This is to avoid memory peak usage when the model is large.
      if (!subgraph_context.has_dynamic_input_shape && disable_cpu_fallback) {
        model_proto.reset();
      }
      auto ov_model = CreateOVModel(std::move(model), session_context_, const_outputs_map_);
      exe_network_ = OVCore::Get()->CompileModel(
          ov_model, hw_target, device_config, enable_causallm, subgraph_context_.subgraph_name);
    }
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
  infer_req_pool_ = std::make_unique<InferRequestPool>(exe_network_, num_infer_req, std::move(initializer));
  bindings_ = std::make_unique<OnnxToOvNetworkBindings>(exe_network_, subgraph_context_, session_context_);
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
  if (session_context_.precision.find("ACCURACY") != std::string::npos) {
    if (session_context_.OpenVINO_Version.at(0) >= 2024) {
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

    if ((session_context_.device_type.find("NPU") != std::string::npos) && session_context_.enable_causallm) {
      if (target_config.find("NPU") != target_config.end()) {
        auto npu_genai_config = target_config.at("NPU");
        CausalLMConfig().ApplyConfig(npu_genai_config, device_config);
      } else {
        LOGS_DEFAULT(WARNING) << "ORT GenAI CausalLMConfig Configuration not found.";
      }
    }

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
    auto find_device_type_mode = [&](const std::string& device_type) -> std::string {
      std::string device_mode = "";
      auto delimiter_pos = device_type.find(':');
      if (delimiter_pos != std::string::npos) {
        std::stringstream str_stream(device_type.substr(0, delimiter_pos));
        std::getline(str_stream, device_mode, ',');
      }
      return device_mode;
    };

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
        if ((key.find("NPUW") != std::string::npos) ||
            ((device_config.find(key) != device_config.end()) && session_context_.enable_causallm)) {
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
      //// Parse to get the device mode (e.g., "AUTO:CPU,GPU" -> "AUTO")
      std::unordered_set<std::string> supported_mode = {"AUTO", "HETERO", "MULTI"};
      auto device_mode = find_device_type_mode(session_context_.device_type);
      ORT_ENFORCE(supported_mode.find(device_mode) != supported_mode.end(), " Invalid device mode is passed : ", session_context_.device_type);
      // Parse individual devices (e.g., "AUTO:CPU,GPU" -> ["CPU", "GPU"])
      auto individual_devices = parse_individual_devices(session_context_.device_type);
      if (!device_mode.empty()) individual_devices.emplace_back(device_mode);

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

void BasicBackend::ValidateOrtDimsAgainstPartialShape(const std::vector<int64_t>& ort_dims,
                                                      const ov::PartialShape& partial_shape) const {
  // Check if the number of dimensions matches
  if (static_cast<int64_t>(ort_dims.size()) != partial_shape.rank().get_length()) {
    ORT_THROW("Mismatch in number of dimensions between ORT tensor and OpenVINO PartialShape.");
  }
  // Validate each dimension
  for (size_t i = 0; i < ort_dims.size(); ++i) {
    const auto& ov_dim = partial_shape[i];  // OpenVINO dimension at index i
    int64_t ort_dim = ort_dims[i];          // ORT dimension at index i

    // Check if the ORT dimension is within the specified range
    int64_t min_dim = ov_dim.get_min_length();
    int64_t max_dim = ov_dim.get_max_length();
    if (ort_dim < min_dim || ort_dim > max_dim) {
      ORT_THROW(" ORT Dimension is out of range");
    }
  }
}

void BasicBackend::RewindKVCache(size_t index) {
  infer_req_pool_->forEachIdleRequest([&](OVInferRequestPtr& infer_request) {
    infer_request->RewindKVCache(index);
  });
}

void BasicBackend::Infer(OrtKernelContext* ctx) const {
  Ort::KernelContext context(ctx);

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";

  if (subgraph_context_.is_constant) {
    for (const auto& item : const_outputs_map_) {
      std::string out_name = item.first;
      std::shared_ptr<ov::Node> node = item.second;
      Ort::UnownedValue output_tensor = GetOutputTensor(context,
                                                        std::move(out_name),
                                                        subgraph_context_.output_names,
                                                        node);
      FillOutputsWithConstantData(std::move(node), output_tensor);
    }

    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";

    if (IsCILogEnabled()) {
      std::cout << "Inference successful" << std::endl;
    }
    return;
  }

  // guarded_request will be released back to the pool when it goes out of scope
  auto guarded_request = infer_req_pool_->getRequest();
  auto& infer_request = guarded_request.infer_request_;

  if (bindings_->has_dynamic_io_) {
    // Dynamic shape inference

    // We don't know the output shapes so we need to get the outputs from the infer request and copy them into the ort
    // tensors instead of binding them to the infer request directly.

    // Bind inputs
    for (const auto& input_info : bindings_->network_inputs_) {
      // Set the input shape based on the input tensor from ort
      auto tensor = context.GetInput(input_info.onnx_index);
      auto ort_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
      if (input_info.IsBoundedDynamic()) {
        ValidateOrtDimsAgainstPartialShape(ort_shape, input_info.shape);
      }
      auto input_shape = ParameterShape(ort_shape);

      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_shape,
                               const_cast<void*>(tensor.GetTensorRawData()));
    }

    // Run Inference
    infer_request->Infer();

    // Copy outputs
    for (const auto& output_info : bindings_->network_outputs_) {
      auto ov_tensor = infer_request->GetTensor(output_info.name);
      auto output_shape = ParameterShape::ToOrtShape(ov_tensor->get_shape());
      auto ort_tensor = context.GetOutput(output_info.onnx_index, output_shape);

      ORT_ENFORCE(ov_tensor->get_byte_size() == ort_tensor.GetTensorSizeInBytes(),
                  log_tag + "Output tensor size mismatch for " + output_info.name);

      std::memcpy(ort_tensor.GetTensorMutableRawData(),
                  ov_tensor->data(),
                  ov_tensor->get_byte_size());
    }
  } else {
    // Static shape inference

    // Bind inputs
    for (const auto& input_info : bindings_->network_inputs_) {
      infer_request->SetTensor(input_info.name,
                               input_info.type,
                               input_info.shape,
                               const_cast<void*>(context.GetInput(input_info.onnx_index).GetTensorRawData()));
    }

    // Bind outputs
    for (const auto& output_info : bindings_->network_outputs_) {
      infer_request->SetTensor(output_info.name,
                               output_info.type,
                               output_info.shape,
                               context.GetOutput(output_info.onnx_index, output_info.shape).GetTensorMutableRawData());
    }

    // Run Inference
    infer_request->Infer();
  }

  // Fill constant outputs if needed
  for (const auto& [name, node] : const_outputs_map_) {
    Ort::UnownedValue output_tensor = GetOutputTensor(context,
                                                      name,
                                                      subgraph_context_.output_names,
                                                      node);
    FillOutputsWithConstantData(node, output_tensor);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
  if (IsCILogEnabled()) {
    std::cout << "Inference successful" << std::endl;
  }

#ifndef NDEBUG
  // Print performance counts before releasing the infer_request for thread safety
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::string& hw_target = session_context_.device_type;
    printPerformanceCounts(infer_request, std::cout, hw_target);
  }
#endif
}

}  // namespace openvino_ep
}  // namespace onnxruntime
