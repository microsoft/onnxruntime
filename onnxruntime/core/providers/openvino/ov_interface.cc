// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_interface.h"

#include <format>

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/providers/openvino/backends/basic_backend.h"
#include "core/providers/openvino/ov_stateful_patch_utils.h"

namespace onnxruntime {
namespace openvino_ep {

template <typename Func, typename... Args>
inline auto OvExceptionBoundary(Func &&func, std::format_string<Args...>&& fmt, Args&&... args) {
  try {
    return func();
  } catch (const ov::Exception& e) {
    ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)) + ": " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + std::vformat(fmt.get(), std::make_format_args(args...)));
  }
}

#ifndef NDEBUG
void printDebugInfo(const ov::CompiledModel& obj) {
  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    // output of the actual settings that the device selected
    auto supported_properties = obj.get_property(ov::supported_properties);
    std::cout << "Model:" << std::endl;
    for (const auto& cfg : supported_properties) {
      if (cfg == ov::supported_properties)
        continue;
      auto prop = obj.get_property(cfg);
      if (cfg == ov::device::properties) {
        auto devices_properties = prop.as<ov::AnyMap>();
        for (auto& item : devices_properties) {
          std::cout << "  " << item.first << ": " << std::endl;
          for (auto& item2 : item.second.as<ov::AnyMap>()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            if (item2.first == ov::supported_properties || item2.first == "SUPPORTED_CONFIG_KEYS)" ||
                item2.first == "SUPPORTED_METRICS")
              continue;
            OPENVINO_SUPPRESS_DEPRECATED_END
            std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
          }
        }
      } else {
        std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
      }
    }
  }
}
#endif

// Function to check if a given OV property is enabled
std::optional<bool> queryOVProperty(const std::string& property, const std::string& device_type) {
  try {
    // Get the property value
    auto supported_properties = OVCore::Get()->core.get_property(device_type, ov::supported_properties);
    return std::find(supported_properties.begin(), supported_properties.end(), property) != supported_properties.end();
  } catch (const std::exception&) {
    return std::nullopt;  // Property not found or invalid
  }
}

std::shared_ptr<OVNetwork> OVCore::ReadModel(std::string&& model, const std::string& model_path) {
  return OvExceptionBoundary([&]() {
    std::istringstream modelStringStream(std::move(model));
    std::istream& modelStream = modelStringStream;
    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream, model_path};

    FE = manager.load_by_model(params);
    if (FE) {
      inputModel = FE->load(params);
      return FE->convert(inputModel);
    } else {
      ORT_THROW(log_tag + "Unknown exception while Reading network");
    }
  },
                             "Exception while Reading network");
}

OVExeNetwork OVCore::StatefulCompileModel(std::shared_ptr<OVNetwork>& model,
                                          std::string& hw_target,
                                          const ov::AnyMap& device_config) {
  ov::CompiledModel compiled_model;
  ov::AnyMap config = device_config;

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateless OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Converting from Stateless OV Model to Stateful OV Model" << std::endl;
  bool model_status = IsStateful(model);
  LOGS_DEFAULT(INFO) << log_tag << "Model IsStateful() Status:\t" << (model_status ? "True" : "False");
  if (!model_status) {
    PatchStatefulDecoder(model);
  }

  if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Stateful OV Model Statistic:" << std::endl;
    LogBasicModelInfo(model);
  }

  auto kv_pos = GetKVAxesPos(model);

  if (hw_target.find("NPU") != std::string::npos) {
    KVDesc kv_desc;
    auto parse_genai_config = [&](const std::string& key, unsigned int default_value) {
      return (config.count(key) && !config.at(key).empty() && config.at(key).as<std::string>() != "0") ? config.at(key).as<unsigned int>() : default_value;
    };

    kv_desc.max_prompt_len = parse_genai_config("MAX_PROMPT_LEN", CausalLMConfig().max_prompt_len);
    kv_desc.min_response_len = parse_genai_config("MIN_RESPONSE_LEN", CausalLMConfig().min_response_len);

    // For compilation, MAX_PROMPT_LEN & MIN_RESPONSE_LEN should not be 0
    if (kv_desc.max_prompt_len == 0 || kv_desc.min_response_len == 0) {
      ORT_THROW(log_tag + "MAX_PROMPT_LEN and MIN_RESPONSE_LEN cannot be 0 or empty");
    }

    if (onnxruntime::openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "kv_pos.batch = " << kv_pos.batch << std::endl;
      std::cout << "kv_pos.seq_len = " << kv_pos.seq_len << std::endl;
      std::cout << "kv_desc.max_prompt_len:\t" << kv_desc.max_prompt_len << std::endl;
      std::cout << "kv_desc.min_response_len:\t" << kv_desc.min_response_len << std::endl;
    }

    UpdateNPUConfig(config, kv_pos, kv_desc);
  } else {
    // This patches the OV IR model so that it only produces the logits required for sampling.
    // Actually either way that happens within NPUW::LLMCompiledModel creation for NPU device,
    // while this is here mostly to align this behavior for other devices viz. (CPU, GPU).
    ApplySliceBeforeMatmulTransformation(model);
  }

  LOGS_DEFAULT(INFO) << log_tag << "Compiling OV Model using Stateful Transformation flow";
  compiled_model = OVCore::Get()->core.compile_model(model, hw_target, config);
  OVExeNetwork exe(compiled_model, hw_target, true);
  return exe;
}

OVExeNetwork OVCore::CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  bool enable_causallm,
                                  const std::string& name) {
  return OvExceptionBoundary([&]() {
    OVExeNetwork exe;
    if (enable_causallm) {
    auto mutable_model = ie_cnn_network->clone();
    exe = OVCore::Get()->StatefulCompileModel(mutable_model, hw_target, device_config);
    } else {
    auto obj = core.compile_model(ie_cnn_network, hw_target, device_config);
    exe = OVExeNetwork(obj, hw_target);
    }

#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif

    return exe;
  },
                             "Exception while Loading Network for graph {}", name);
}

OVExeNetwork OVCore::CompileModel(const std::string& onnx_model,
                                  std::string& hw_target,
                                  ov::AnyMap& device_config,
                                  const std::string& name) {
  return OvExceptionBoundary([&]() {
    ov::CompiledModel obj;

    obj = core.compile_model(onnx_model, ov::Tensor(), hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(obj);
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  },
                             "Exception while Loading Network for graph {}", name);
}

OVExeNetwork OVCore::ImportModel(std::istream& model_stream,
                                 std::string hw_target,
                                 const ov::AnyMap& device_config,
                                 std::string name) {
  return OvExceptionBoundary([&]() {
    ov::CompiledModel obj;
    obj = core.import_model(model_stream, hw_target, device_config);
#ifndef NDEBUG
    printDebugInfo(exe.Get());
#endif
    OVExeNetwork exe(obj, hw_target);
    return exe;
  },
                             "Exception while Loading Network for graph {}", name);
}

void OVCore::SetCache(const std::string& cache_dir_path) {
  core.set_property(ov::cache_dir(cache_dir_path));
}

std::vector<std::string> OVCore::GetAvailableDevices() const {
  std::vector<std::string> available_devices = core.get_available_devices();
  return available_devices;
}

std::vector<std::string> OVCore::GetAvailableDevices(const std::string& device_type) const {
  std::vector<std::string> available_devices;
  std::vector<std::string> devicesIDs;
  // Uses logic from OpenVINO to only return available devices of the specified type (e.g. CPU, NPU or GPU)
  try {
    devicesIDs = core.get_property(device_type, ov::available_devices);
  } catch (const ov::Exception&) {
    // plugin is not created by e.g. invalid env
    // Empty device list will be returned
  } catch (const std::exception& ex) {
    ORT_THROW(log_tag + "An exception occurred while trying to create the ",
              device_type,
              " device: ",
              ex.what());
  } catch (...) {
    ORT_THROW(log_tag + "Unknown exception occurred while trying to create the ",
              device_type,
              " device");
  }

  if (devicesIDs.size() > 1 ||
      (devicesIDs.size() == 1 && devicesIDs[0] == "0")) {
    for (const auto& deviceID : devicesIDs) {
      available_devices.push_back(device_type + '.' + deviceID);
    }
  }
  if (!devicesIDs.empty()) {
    available_devices.push_back(device_type);
  }

  return available_devices;
}

void OVCore::SetStreams(const std::string& device_type, int num_streams) {
  core.set_property(device_type, {ov::num_streams(num_streams)});
}

std::shared_ptr<OVInferRequest> OVExeNetwork::CreateInferRequest() {
   return OvExceptionBoundary([&]() {
    auto infReq = compiled_model_obj.create_infer_request();
    std::shared_ptr<OVInferRequest> ovInfReq;
    if (is_stateful_causallm) {
      ovInfReq = std::make_shared<StatefulOVInferRequest>(std::move(infReq), target_device);
    } else {
      ovInfReq = std::make_shared<OVInferRequest>(std::move(infReq));
    }
    return ovInfReq;
  },

                             "Exception while creating InferRequest object");
}

OVTensorPtr OVInferRequest::GetTensor(const std::string& input_name) {
  return OvExceptionBoundary([&]() {
    auto tobj = ovInfReq.get_tensor(input_name);
    OVTensorPtr blob = std::make_shared<OVTensor>(tobj);
    return blob;
  },
                             " Cannot access IE Blob for input: {}", input_name);
}

std::string OVInferRequest::GetInputTensorName(uint32_t index) {
  return OvExceptionBoundary([&]() {
    const auto& model = ovInfReq.get_compiled_model();
    return *model.input(index).get_names().begin();
  },
                             " Cannot access IE Blob for input number: {}", index);
}

void OVInferRequest::SetTensor(const std::string& name, OVTensorPtr& blob) {
  OvExceptionBoundary([&]() {
    ovInfReq.set_tensor(name, *(blob.get()));
  },
                      " Cannot set Remote Blob for output: {}", name);
}

uint32_t OVInferRequest::GetNumInputs() {
  return static_cast<uint32_t>(ovInfReq.get_compiled_model().inputs().size());
}

void OVInferRequest::Infer() {
  OvExceptionBoundary([&]() {
    ovInfReq.infer();
  },
                      "In Error Couldn't start Inference");
}

StatefulOVInferRequest::StatefulOVInferRequest(ov::InferRequest infer_request, std::string device)
    : OVInferRequest(std::move(infer_request)), target_device(device) {
  bool gpu_or_npu = ((device.find("NPU") != std::string::npos) || (device.find("GPU") != std::string::npos));
  if (gpu_or_npu) {
    prefill_use_full_chat_history = true;
  }
}

void StatefulOVInferRequest::FillTensor(const std::string& tensor_name, const ov::element::Type& type,
                                        const std::vector<size_t>& shape, int32_t fill_value) {
  ov::Tensor tensor = ov::Tensor(type, shape);
  std::fill_n(tensor.data<int32_t>(), tensor.get_size(), fill_value);
  ovInfReq.set_tensor(tensor_name, tensor);
}

void StatefulOVInferRequest::CacheTensor(const std::string& tensor_name, std::vector<int64_t>& cache) {
  auto tensor = ovInfReq.get_tensor(tensor_name);
  auto* pData = tensor.data<int64_t>();
  for (size_t i = 0; i < tensor.get_size(); i++) {
    cache.emplace_back(pData[i]);
  }
}

void StatefulOVInferRequest::SetTensorFromCache(const std::string& tensor_name,
                                                const std::vector<int64_t>& cache_data) {
  auto tensor = ovInfReq.get_tensor(tensor_name);
  auto new_shape = tensor.get_shape();
  new_shape[1] = cache_data.size();

  auto new_tensor = ov::Tensor(tensor.get_element_type(), new_shape);
  auto* pNewData = new_tensor.data<int64_t>();
  std::memcpy(pNewData, cache_data.data(), cache_data.size() * sizeof(int64_t));

  ovInfReq.set_tensor(tensor_name, new_tensor);
}

std::optional<ov::Tensor> StatefulOVInferRequest::FindTensor(const std::string& tensor_name) {
  // Check if tensor exists by examining input names in the compiled model
  const auto& model = ovInfReq.get_compiled_model();
  bool tensor_exists = false;

  for (const auto& input : model.inputs()) {
    const auto& names = input.get_names();
    if (names.find(tensor_name) != names.end()) {
      tensor_exists = true;
      break;
    }
  }

  if (tensor_exists) {
    return ovInfReq.get_tensor(tensor_name);
  }

  return std::nullopt;
}

void StatefulOVInferRequest::PreProcessInferRequest() {
  // Workaround: Setting the value here as it cannot be set at the ORT GenAI layer currently.
  // TODO(ankit): Address this issue and implement the fix at the appropriate layer.
  FillTensor("beam_idx", ov::element::i32, {1}, 0);

  // If 'prefill use full chat history' mode is enabled, we need to cache input_ids and position_ids.
  if (prefill_use_full_chat_history) {
    auto input_ids_tensor = ovInfReq.get_tensor("input_ids");
    CacheTensor("input_ids", cached_input_ids);

    // "position_ids" (GQA with Rotary Embeddings doesnt have position_ids) - check if exists
    auto position_ids_opt = FindTensor("position_ids");
    bool has_position_ids = position_ids_opt.has_value();

    if (has_position_ids) {
      CacheTensor("position_ids", cached_position_ids);
    }

    // If we're about to run the prefill model
    if (input_ids_tensor.get_size() > 1) {
      // Check if the size of the current "input_ids" tensor does not match the size of the cached "input_ids".
      // This indicates that we are running a subsequent prompt (not the initial prefill).
      if (input_ids_tensor.get_shape()[1] != cached_input_ids.size()) {
        // Clear the internal KVCache state. For NPU device, this operation is a no-op.
        ovInfReq.reset_state();

        // Set tensors using cached values
        SetTensorFromCache("input_ids", cached_input_ids);

        // Only set position_ids if it exists and we have cached values
        if (has_position_ids && !cached_position_ids.empty()) {
          SetTensorFromCache("position_ids", cached_position_ids);
        }
      }
    }
  }
}

void StatefulOVInferRequest::Infer() {
  PreProcessInferRequest();
  OVInferRequest::Infer();
}

void StatefulOVInferRequest::RewindKVCache(size_t index) {
  LOGS_DEFAULT(INFO) << log_tag << "RewindKVCache: Rewinding OpenVINO-internal KVCache state to index=" << index;

  if (prefill_use_full_chat_history) {
    // Clear the internal KVCache state. For NPU device, this operation is a no-op.
    ovInfReq.reset_state();

    // Resize the cached "input_ids" and "position_ids" to the specified index.
    if (cached_input_ids.size() > index) {
      cached_input_ids.resize(index);
    }

    if (cached_position_ids.size() > index) {
      cached_position_ids.resize(index);
    }
  } else {
    if (index == 0) {
      // In this case, since we're resetting the entire KVCache, simply reset the state.
      ovInfReq.reset_state();
    } else {
      // Retrieve KVCache states and trim them to the specified index.
      // The following logic is adapted from:
      // https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/1/src/cpp/src/utils.cpp#L329
      auto states = ovInfReq.query_state();
      for (auto& state : states) {
        ov::Tensor old_tensor = state.get_state();
        // Tensor shape: [batch_size, num_kv_heads, seq_len, head_size]
        auto shape = old_tensor.get_shape();

        if (shape[2] > index) {
          // Update the sequence length dimension to the specified index.
          shape[2] = index;

          ov::Coordinate new_shape_begin{0, 0, 0, 0};
          ov::Coordinate new_shape_end{shape};

          // Create a trimmed tensor with the updated shape.
          auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

          // Copy the trimmed tensor into a new tensor and update the state.
          ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
          trimmed_tensor.copy_to(new_tensor);

          state.set_state(new_tensor);
        }
      }
    }
  }
}
}  // namespace openvino_ep
}  // namespace onnxruntime
