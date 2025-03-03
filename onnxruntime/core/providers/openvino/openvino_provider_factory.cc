// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <map>
#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace openvino_ep {
void ParseConfigOptions(ProviderInfo& pi, const ConfigOptions& config_options) {
  pi.so_disable_cpu_ep_fallback = config_options.GetConfigOrDefault(kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
  pi.so_context_enable = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  pi.so_context_embed_mode = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
  pi.so_share_ep_contexts = config_options.GetConfigOrDefault(kOrtSessionOptionShareEpContexts, "0") == "1";
  pi.so_context_file_path = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
}

void* ParseUint64(const ProviderOptions& provider_options, std::string option_name) {
  if (provider_options.contains(option_name)) {
    uint64_t number = std::strtoull(provider_options.at(option_name).data(), nullptr, 16);
    return reinterpret_cast<void*>(number);
  } else {
    return nullptr;
  }
}

bool ParseBooleanOption(const ProviderOptions& provider_options, std::string option_name) {
  if (provider_options.contains(option_name)) {
    const auto& value = provider_options.at(option_name);
    if (value == "true" || value == "True") {
      return true;
    } else if (value == "false" || value == "False") {
      return false;
    } else {
      ORT_THROW("[ERROR] [OpenVINO-EP] ", option_name, " should be a boolean.\n");
    }
  }
  return false;
}

std::string ParseDeviceType(std::shared_ptr<OVCore> ov_core, const ProviderOptions& provider_options, std::string option_name) {
  const std::vector<std::string> ov_available_devices = ov_core->GetAvailableDevices();

  std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                     "GPU.0", "GPU.1", "NPU"};
  std::set<std::string> deprecated_device_types = {"CPU_FP32", "GPU_FP32",
                                                   "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                   "GPU.0_FP16", "GPU.1_FP16"};

  // Expand set of supported device with OV devices
  ov_supported_device_types.insert(ov_available_devices.begin(), ov_available_devices.end());

  if (provider_options.contains(option_name)) {
    const auto& selected_device = provider_options.at("device_type");

    if (deprecated_device_types.contains(selected_device)) {
      // Deprecated device and precision is handled together at ParsePrecision
      return selected_device;
    }

    if (!((ov_supported_device_types.contains(selected_device)) ||
          (selected_device.find("HETERO:") == 0) ||
          (selected_device.find("MULTI:") == 0) ||
          (selected_device.find("AUTO:") == 0))) {
      ORT_THROW(
          "[ERROR] [OpenVINO] You have selected wrong configuration value for the key 'device_type'. "
          "Select from 'CPU', 'GPU', 'NPU', 'GPU.x' where x = 0,1,2 and so on or from"
          " HETERO/MULTI/AUTO options available. \n");
    }
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
  } else {
    std::string default_device;

    // Take default behavior from project configuration
#if defined OPENVINO_CONFIG_CPU
    default_device = "CPU";
#elif defined OPENVINO_CONFIG_GPU
    default_device = "GPU";
#elif defined OPENVINO_CONFIG_NPU
    default_device = "NPU";
#elif defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
    default_device = DEVICE_NAME;

    // Validate that devices passed are valid
    int delimit = device_type.find(":");
    const auto& devices = device_type.substr(delimit + 1);
    auto device_list = split(devices, ',');
    for (const auto& device : devices) {
      if (!ov_supported_device_types.contains(device)) {
        ORT_THROW("[ERROR] [OpenVINO] Invalid device selected: ", device);
      }
    }
#endif

    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << default_device;
    return default_device;
  }
}

// Depends on ProviderOptions.
std::string ParsePrecision(const ProviderOptions& provider_options, std::string& device_type, const std::string& option_name) {
  using DeviceName = std::string;
  using DefaultValue = std::string;
  using ValidValues = std::list<std::string>;
  using foo = std::pair<DefaultValue, ValidValues>;
  using ParserHelper = std::map<DeviceName, foo>;
  ParserHelper helper = {
      {"GPU", {"FP16", {"FP16", "FP32"}}},
      {"NPU", {"FP16", {"FP16"}}},
      {"CPU", {"FP32", {"FP32"}}},
  };

  std::set<std::string> deprecated_device_types = {"CPU_FP32", "GPU_FP32",
                                                   "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                   "GPU.0_FP16", "GPU.1_FP16"};

  if (provider_options.contains(option_name)) {
    // Start by checking if the device_type is a normal valid one
    if (helper.contains(device_type)) {
      auto const& valid_values = helper[device_type].second;
      const auto& precision = provider_options.at(option_name);
      if (precision == "ACCURACY") {
        return valid_values.back();  // Return highest supported precision
      } else {
        if (std::find(valid_values.begin(), valid_values.end(), precision) != valid_values.end()) {
          return precision;  // Return precision selected if valid
        } else {
          auto value_iter = valid_values.begin();
          std::string valid_values_joined = *value_iter;
          // Append 2nd and up, if only one then ++value_iter is same as end()
          for (++value_iter; value_iter != valid_values.end(); ++value_iter) {
            valid_values_joined += ", " + *value_iter;
          }

          ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. ", device_type, " only supports", valid_values_joined, ".\n");
        }
      }
    } else if (deprecated_device_types.contains(device_type)) {
      LOGS_DEFAULT(WARNING) << "[OpenVINO] Selected 'device_type' " + device_type + " is deprecated. \n"
                            << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
                            << "'GPU.1', 'NPU' or from"
                            << " HETERO/MULTI/AUTO options and set 'precision' separately. \n";
      auto delimit = device_type.find("_");
      device_type = device_type.substr(0, delimit);
      return device_type.substr(delimit + 1);
    }
  }
  // Return default
  return helper[device_type].first;
}

void ParseProviderOptions([[maybe_unused]] ProviderInfo& result, [[maybe_unused]] const ProviderOptions& config_options) {}

struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(ProviderInfo provider_info, std::shared_ptr<SharedContext> shared_context)
      : provider_info_(std::move(provider_info)), shared_context_(shared_context) {}

  ~OpenVINOProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<OpenVINOExecutionProvider>(provider_info_, shared_context_);
  }

 private:
  ProviderInfo provider_info_;
  std::shared_ptr<SharedContext> shared_context_;
};

struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    return OVCore::Get()->GetAvailableDevices();
  }
};

struct OpenVINO_Provider : Provider {
  void* GetInfo() override { return &info_; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    // Extract the void_params into ProviderOptions and ConfigOptions
    using ConfigBuffer = std::pair<const ProviderOptions*, const ConfigOptions&>;
    const ConfigBuffer* buffer = reinterpret_cast<const ConfigBuffer*>(void_params);
    const auto& provider_options = *buffer->first;
    const auto& config_options = buffer->second;

    ProviderInfo pi;

    std::string bool_flag = "";

    // Minor optimization: we'll hold an OVCore reference to ensure we don't create a new core between ParseDeviceType and
    // (potential) SharedContext creation.
    auto ov_core = OVCore::Get();
    pi.device_type = ParseDeviceType(ov_core, provider_options, "device_type");

    if (provider_options.contains("device_id")) {
      std::string dev_id = provider_options.at("device_id").data();
      LOGS_DEFAULT(WARNING) << "[OpenVINO] The options 'device_id' is deprecated. "
                            << "Upgrade to set deice_type and precision session options.\n";
      if (dev_id == "CPU" || dev_id == "GPU" || dev_id == "NPU") {
        pi.device_type = std::move(dev_id);
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
      }
    }
    if (provider_options.contains("cache_dir")) {
      pi.cache_dir = provider_options.at("cache_dir");
    }

    pi.precision = ParsePrecision(provider_options, pi.device_type, "precision");

    if (provider_options.contains("load_config")) {
      auto parse_config = [&](const std::string& config_str) -> std::map<std::string, ov::AnyMap> {
        // If the config string is empty, return an empty map and skip processing
        if (config_str.empty()) {
          LOGS_DEFAULT(WARNING) << "Empty OV Config Map passed. Skipping load_config option parsing.\n";
          return {};
        }

        std::stringstream input_str_stream(config_str);
        std::map<std::string, ov::AnyMap> target_map;

        try {
          nlohmann::json json_config = nlohmann::json::parse(input_str_stream);

          if (!json_config.is_object()) {
            ORT_THROW("Invalid JSON structure: Expected an object at the root.");
          }

          for (auto& [key, value] : json_config.items()) {
            ov::AnyMap inner_map;

            // Ensure the key is one of "CPU", "GPU", or "NPU"
            if (key != "CPU" && key != "GPU" && key != "NPU") {
              LOGS_DEFAULT(WARNING) << "Unsupported device key: " << key << ". Skipping entry.\n";
              continue;
            }

            // Ensure that the value for each device is an object (PROPERTY -> VALUE)
            if (!value.is_object()) {
              ORT_THROW("Invalid JSON structure: Expected an object for device properties.");
            }

            for (auto& [inner_key, inner_value] : value.items()) {
              if (inner_value.is_string()) {
                inner_map[inner_key] = inner_value.get<std::string>();
              } else if (inner_value.is_number_integer()) {
                inner_map[inner_key] = inner_value.get<int64_t>();
              } else if (inner_value.is_number_float()) {
                inner_map[inner_key] = inner_value.get<double>();
              } else if (inner_value.is_boolean()) {
                inner_map[inner_key] = inner_value.get<bool>();
              } else {
                LOGS_DEFAULT(WARNING) << "Unsupported JSON value type for key: " << inner_key << ". Skipping key.";
              }
            }
            target_map[key] = std::move(inner_map);
          }
        } catch (const nlohmann::json::parse_error& e) {
          // Handle syntax errors in JSON
          ORT_THROW("JSON parsing error: " + std::string(e.what()));
        } catch (const nlohmann::json::type_error& e) {
          // Handle invalid type accesses
          ORT_THROW("JSON type error: " + std::string(e.what()));
        } catch (const std::exception& e) {
          ORT_THROW("Error parsing load_config Map: " + std::string(e.what()));
        }
        return target_map;
      };

      pi.load_config = parse_config(provider_options.at("load_config"));
    }

    pi.context = ParseUint64(provider_options, "context");
#if defined(IO_BUFFER_ENABLED)
    // a valid context must be provided to enable IO Buffer optimizations
    if (pi.context == nullptr) {
#undef IO_BUFFER_ENABLED
#define IO_BUFFER_ENABLED = 0
      LOGS_DEFAULT(WARNING) << "Context is not set. Disabling IO Buffer optimization";
    }
#endif

    if (provider_options.contains("num_of_threads")) {
      if (!std::all_of(provider_options.at("num_of_threads").begin(),
                       provider_options.at("num_of_threads").end(), ::isdigit)) {
        ORT_THROW("[ERROR] [OpenVINO-EP] Number of threads should be a number. \n");
      }
      pi.num_of_threads = std::stoi(provider_options.at("num_of_threads"));
      if (pi.num_of_threads <= 0) {
        pi.num_of_threads = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_threads' should be in the positive range.\n "
                              << "Executing with num_threads=1";
      }
    }

    if (provider_options.contains("model_priority")) {
      pi.model_priority = provider_options.at("model_priority").data();
      std::vector<std::string> supported_priorities({"LOW", "MEDIUM", "HIGH", "DEFAULT"});
      if (std::find(supported_priorities.begin(), supported_priorities.end(),
                    pi.model_priority) == supported_priorities.end()) {
        pi.model_priority = "DEFAULT";
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'model_priority' "
                              << "is not one of LOW, MEDIUM, HIGH, DEFAULT. "
                              << "Executing with model_priorty=DEFAULT";
      }
    }

    if (provider_options.contains("num_streams")) {
      pi.num_streams = std::stoi(provider_options.at("num_streams"));
      if (pi.num_streams <= 0) {
        pi.num_streams = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_streams' should be in the range of 1-8.\n "
                              << "Executing with num_streams=1";
      }
    }
    pi.enable_opencl_throttling = ParseBooleanOption(provider_options, "enable_opencl_throttling");

    pi.enable_qdq_optimizer = ParseBooleanOption(provider_options, "enable_qdq_optimizer");

    pi.disable_dynamic_shapes = ParseBooleanOption(provider_options, "disable_dynamic_shapes");

    ParseConfigOptions(pi, config_options);

    // Always true for NPU plugin or when passed .
    if (pi.device_type.find("NPU") != std::string::npos) {
      pi.disable_dynamic_shapes = true;
    }

    // Append values to config to support weight-as-inputs conversion for shared contexts
    if (pi.so_share_ep_contexts) {
      ov::AnyMap map;
      map["NPU_COMPILATION_MODE_PARAMS"] = "enable-wd-blockarg-input=true compute-layers-with-higher-precision=Sqrt,Power,ReduceSum";
      pi.load_config["NPU"] = std::move(map);
    }

    return std::make_shared<OpenVINOProviderFactory>(pi, SharedContext::Get());
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }

 private:
  ProviderInfo_OpenVINO_Impl info_;
};  // OpenVINO_Provider

}  // namespace openvino_ep
}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  static onnxruntime::openvino_ep::OpenVINO_Provider g_provider;
  return &g_provider;
}
}
