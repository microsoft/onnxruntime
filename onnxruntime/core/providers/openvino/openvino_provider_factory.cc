// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"
#include "core/providers/openvino/openvino_parser_utils.h"

namespace onnxruntime {
namespace openvino_ep {
void ParseConfigOptions(ProviderInfo& pi) {
  if (pi.config_options == nullptr)
    return;

  pi.so_disable_cpu_ep_fallback = pi.config_options->GetConfigOrDefault(kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
  pi.so_context_enable = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  pi.so_context_embed_mode = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
  pi.so_share_ep_contexts = pi.config_options->GetConfigOrDefault(kOrtSessionOptionShareEpContexts, "0") == "1";
  pi.so_context_file_path = pi.config_options->GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");

  if (pi.so_share_ep_contexts) {
    ov::AnyMap map;
    map["NPU_COMPILATION_MODE_PARAMS"] = "enable-wd-blockarg-input=true compute-layers-with-higher-precision=Sqrt,Power,ReduceSum";
    pi.load_config["NPU"] = std::move(map);
  }
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

std::string ParseDeviceType(std::shared_ptr<OVCore> ov_core, const ProviderOptions& provider_options) {
  std::set<std::string> supported_device_types = {"CPU", "GPU", "NPU"};
  std::set<std::string> supported_device_modes = {"AUTO", "HETERO", "MULTI"};
  std::vector<std::string> devices_to_check;
  std::string selected_device;
  std::vector<std::string> luid_list;
  std::string device_mode = "";
  std::map<std::string, std::string> ov_luid_map;

  if (provider_options.contains("device_type")) {
    selected_device = provider_options.at("device_type");
    std::erase(selected_device, ' ');
    if (selected_device == "AUTO") return selected_device;

    if (auto delimit = selected_device.find(":"); delimit != std::string::npos) {
      device_mode = selected_device.substr(0, delimit);
      if (supported_device_modes.contains(device_mode)) {
        const auto& devices = selected_device.substr(delimit + 1);
        devices_to_check = split(devices, ',');
        ORT_ENFORCE(devices_to_check.size() > 0, "Mode AUTO/HETERO/MULTI should have devices listed based on priority");
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Invalid device_type is selected. Supported modes are AUTO/HETERO/MULTI");
      }
    } else {
      devices_to_check.push_back(selected_device);
    }
  } else {
    // Take default behavior from project configuration
#if defined OPENVINO_CONFIG_CPU
    selected_device = "CPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_GPU
    selected_device = "GPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_NPU
    selected_device = "NPU";
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
#elif defined OPENVINO_CONFIG_HETERO || defined OPENVINO_CONFIG_MULTI || defined OPENVINO_CONFIG_AUTO
    selected_device = DEVICE_NAME;

    // Add sub-devices to check-list
    int delimit = selected_device.find(":");
    const auto& devices = selected_device.substr(delimit + 1);
    devices_to_check = split(devices, ',');
#endif
  }

  // Get the LUID passed from the provider option in a comma separated string list
  // Compare each of the LUID's against the LUID obtained using ov property and map with the right device
  if (provider_options.contains("device_luid")) {
    std::string luid_str = provider_options.at("device_luid");
    std::erase(luid_str, ' ');
    luid_list = split(luid_str, ',');
  }

  bool all_devices_found = true;

  for (auto device : devices_to_check) {
    bool device_found = false;
    // Check deprecated device format (CPU_FP32, GPU.0_FP16, etc.) and remove the suffix in place
    // Suffix will be parsed in ParsePrecision
    if (auto delimit = device.find("_"); delimit != std::string::npos) {
      device = device.substr(0, delimit);
    }
    // Just the device name without .0, .1, etc. suffix
    auto device_prefix = device;
    // Check if device index is appended (.0, .1, etc.), if so, remove it
    if (auto delimit = device_prefix.find("."); delimit != std::string::npos)
      device_prefix = device_prefix.substr(0, delimit);
    if (supported_device_types.contains(device_prefix)) {
      try {
        std::vector<std::string> available_devices = ov_core->GetAvailableDevices(device_prefix);
        // Here we need to find the full device name (with .idx, but without _precision)
        if (std::find(std::begin(available_devices), std::end(available_devices), device) != std::end(available_devices))
          device_found = true;
        if (device_prefix != "CPU" && luid_list.size() > 0) {
          for (const auto& dev : available_devices) {
            ov::device::LUID ov_luid = OVCore::Get()->core.get_property(dev, ov::device::luid);
            std::stringstream ov_luid_str;
            ov_luid_str << ov_luid;
            ov_luid_map.emplace(ov_luid_str.str(), dev);
          }
        }
      } catch (const char* msg) {
        ORT_THROW(msg);
      }
    }
    all_devices_found = all_devices_found && device_found;
  }
  if (luid_list.size() > 0) {
    std::string ov_luid_devices;
    for (const auto& luid_str : luid_list) {
      if (ov_luid_map.contains(luid_str)) {
        std::string ov_dev = ov_luid_map.at(luid_str);
        std::string ov_dev_strip = split(ov_dev, '.')[0];
        if (std::find(std::begin(devices_to_check), std::end(devices_to_check), ov_dev) != std::end(devices_to_check) ||
            std::find(std::begin(devices_to_check), std::end(devices_to_check), ov_dev_strip) != std::end(devices_to_check)) {
          if (!ov_luid_devices.empty()) ov_luid_devices = ov_luid_devices + ",";
          ov_luid_devices = ov_luid_devices + ov_dev;
        } else {
          ORT_THROW(" LUID : ", ov_dev, " does not match with device_type : ", selected_device);
        }
      } else {
        ORT_THROW(provider_options.at("device_luid"), " does not exist for the selected device_type : ", selected_device);
      }
    }
    if (!device_mode.empty()) {
      selected_device = device_mode + ":" + ov_luid_devices;
      for (const auto& dev_str : devices_to_check) {
        const auto default_dev = split(dev_str, '.')[0];

        if (ov_luid_devices.find(default_dev) == std::string::npos)
          selected_device = selected_device + "," + dev_str;
      }
    } else {
      selected_device = std::move(ov_luid_devices);
    }
  }
  // If invalid device is chosen error is thrown
  if (!all_devices_found) {
    ORT_THROW(
        "[ERROR] [OpenVINO] You have selected wrong configuration value for the key 'device_type'. "
        "Select from 'CPU', 'GPU', 'NPU', 'GPU.x' where x = 0,1,2 and so on or from"
        " HETERO/MULTI/AUTO/BATCH options available. \n");
  } else {
    LOGS_DEFAULT(INFO) << "[OpenVINO-EP] Choosing Device: " << selected_device;
    return selected_device;
  }
}

void ParseProviderOptions([[maybe_unused]] ProviderInfo& result, [[maybe_unused]] const ProviderOptions& config_options) {}

// Initializes a ProviderInfo struct from a ProviderOptions map and a ConfigOptions map.
static void ParseProviderInfo(const ProviderOptions& provider_options,
                              const ConfigOptions* config_options,
                              /*output*/ ProviderInfo& pi) {
  pi.config_options = config_options;

  // Lambda function to check for invalid keys and throw an error
  auto validateKeys = [&]() {
    for (const auto& pair : provider_options) {
      if (pi.valid_provider_keys.find(pair.first) == pi.valid_provider_keys.end()) {
        ORT_THROW("Invalid provider_option key: " + pair.first);
      }
    }
  };
  validateKeys();

  std::string bool_flag = "";

  // Minor optimization: we'll hold an OVCore reference to ensure we don't create a new core between ParseDeviceType and
  // (potential) SharedContext creation.
  auto ov_core = OVCore::Get();
  pi.device_type = ParseDeviceType(std::move(ov_core), provider_options);

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

  pi.precision = OpenVINOParserUtils::ParsePrecision(provider_options, pi.device_type, "precision");

  if (provider_options.contains("reshape_input")) {
    pi.reshape = OpenVINOParserUtils::ParseInputShape(provider_options.at("reshape_input"));
  }

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
          std::set<std::string> valid_ov_devices = {"CPU", "GPU", "NPU", "AUTO", "HETERO", "MULTI"};
          // Ensure the key is one of "CPU", "GPU", or "NPU"
          if (valid_ov_devices.find(key) == valid_ov_devices.end()) {
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
  try {
    pi.enable_opencl_throttling = ParseBooleanOption(provider_options, "enable_opencl_throttling");

    pi.enable_qdq_optimizer = ParseBooleanOption(provider_options, "enable_qdq_optimizer");

    pi.enable_causallm = ParseBooleanOption(provider_options, "enable_causallm");

    pi.disable_dynamic_shapes = ParseBooleanOption(provider_options, "disable_dynamic_shapes");
  } catch (std::string msg) {
    ORT_THROW(msg);
  }

  if (pi.device_type.find("NPU") != std::string::npos) {
    // For Stateful Compilation i.e. enable_causallm as True, we use the dynamic shapes path.
    if (pi.enable_causallm) {
      pi.disable_dynamic_shapes = false;
    } else {
      pi.disable_dynamic_shapes = true;
    }
  }
}

struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(ProviderInfo provider_info, std::shared_ptr<SharedContext> shared_context)
      : provider_info_(std::move(provider_info)), shared_context_(std::move(shared_context)) {}

  ~OpenVINOProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ParseConfigOptions(provider_info_);
    return std::make_unique<OpenVINOExecutionProvider>(provider_info_, shared_context_);
  }

  // Called by InferenceSession when registering EPs. Allows creation of an EP instance that is initialized with
  // session-level configurations.
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // Extract those EP options into a new "provider_options" map.
    std::string lowercase_ep_name = kOpenVINOExecutionProvider;
    std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    std::string key_prefix = "ep.";
    key_prefix += lowercase_ep_name;
    key_prefix += ".";

    std::unordered_map<std::string, std::string> provider_options;
    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    ProviderInfo provider_info = provider_info_;
    ParseProviderInfo(provider_options, &config_options, provider_info);
    ParseConfigOptions(provider_info);

    auto ov_ep = std::make_unique<OpenVINOExecutionProvider>(provider_info, shared_context_);
    ov_ep->SetLogger(reinterpret_cast<const logging::Logger*>(&session_logger));
    return ov_ep;
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
    if (void_params == nullptr) {
      LOGS_DEFAULT(ERROR) << "[OpenVINO EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<void*, 2> pointers_array = *reinterpret_cast<const std::array<void*, 2>*>(void_params);
    const ProviderOptions provider_options = *reinterpret_cast<ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<ConfigOptions*>(pointers_array[1]);

    ProviderInfo pi;
    ParseProviderInfo(provider_options, config_options, pi);

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
