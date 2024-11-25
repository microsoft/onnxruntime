// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <map>
#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const std::string& device_type, const std::string& precision,
                          size_t num_of_threads,
                          const std::map<std::string, ov::AnyMap>& load_config, const std::string& cache_dir,
                          const std::string& model_priority, int num_streams, void* context,
                          bool enable_opencl_throttling, bool disable_dynamic_shapes,
                          bool enable_qdq_optimizer, const ConfigOptions& config_options)
      : device_type_(device_type),
        precision_(precision),
        num_of_threads_(num_of_threads),
        load_config_(load_config),
        cache_dir_(cache_dir),
        model_priority_(model_priority),
        num_streams_(num_streams),
        context_(context),
        enable_opencl_throttling_(enable_opencl_throttling),
        disable_dynamic_shapes_(disable_dynamic_shapes),
        enable_qdq_optimizer_(enable_qdq_optimizer),
        config_options_(config_options) {}

  ~OpenVINOProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  std::string precision_;
  size_t num_of_threads_;
  const std::map<std::string, ov::AnyMap> load_config_;
  std::string cache_dir_;
  std::string model_priority_;
  int num_streams_;
  void* context_;
  bool enable_opencl_throttling_;
  bool disable_dynamic_shapes_;
  bool enable_qdq_optimizer_;
  const ConfigOptions& config_options_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  bool so_disable_cpu_fallback = config_options_.GetConfigOrDefault(kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
  bool so_export_ep_ctx_blob = config_options_.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";
  bool so_epctx_embed_mode = config_options_.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "1") == "1";
  std::string so_cache_path = config_options_.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "").c_str();

  if (so_export_ep_ctx_blob && !so_cache_path.empty()) {
    cache_dir_ = so_cache_path;
    auto file_path = std::filesystem::path(cache_dir_);
    // ep_context_file_path_ file extension must be .onnx
    if (file_path.extension().generic_string() == ".onnx") {
      // ep_context_file_path_ must be provided as a directory, create it if doesn't exist
      auto parent_path = file_path.parent_path();
      if (!parent_path.empty() && !std::filesystem::is_directory(parent_path) &&
          !std::filesystem::create_directory(parent_path)) {
        ORT_THROW("[ERROR] [OpenVINO] Failed to create directory : " +
                  file_path.parent_path().generic_string() + " \n");
      }
    } else {
      ORT_THROW("[ERROR] [OpenVINO] Invalid ep_ctx_file_path" + cache_dir_ + " \n");
    }
  }

  OpenVINOExecutionProviderInfo info(device_type_, precision_, num_of_threads_, load_config_,
                                     cache_dir_, model_priority_, num_streams_, context_, enable_opencl_throttling_,
                                     disable_dynamic_shapes_, so_export_ep_ctx_blob, enable_qdq_optimizer_,
                                     so_disable_cpu_fallback, so_epctx_embed_mode);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

}  // namespace onnxruntime

namespace onnxruntime {
struct ProviderInfo_OpenVINO_Impl : ProviderInfo_OpenVINO {
  std::vector<std::string> GetAvailableDevices() const override {
    openvino_ep::OVCore ie_core;
    return ie_core.GetAvailableDevices();
  }
} g_info;

struct OpenVINO_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    // Extract the void_params into ProviderOptions and ConfigOptions
    typedef std::pair<const ProviderOptions*, const ConfigOptions&> ConfigBuffer;
    const ConfigBuffer* buffer = reinterpret_cast<const ConfigBuffer*>(void_params);
    auto& provider_options_map = *buffer->first;
    const ConfigOptions& config_options = buffer->second;

    std::string device_type = "";                   // [device_type]: Overrides the accelerator hardware type and
                                                    // precision with these values at runtime.
    std::string precision = "";                     // [precision]: Sets the inference precision for execution.
                                                    // Supported precision for devices are
                                                    // CPU=FP32, GPU=FP32,FP16, NPU=FP16.
                                                    // Not setting precision will execute with optimized precision for
                                                    // best inference latency. set Precision=ACCURACY for executing
                                                    // models with input precision for best accuracy.
    int num_of_threads = 0;                         // [num_of_threads]: Overrides the accelerator default value of
                                                    // number of threads with this value at runtime.
    std::map<std::string, ov::AnyMap> load_config;  // JSON config map to load custom OV parameters.
    std::string cache_dir = "";                     // [cache_dir]: specify the path to
                                                    // dump and load the blobs for the model caching/kernel caching
                                                    // (GPU) feature. If blob files are already present,
                                                    // it will be directly loaded.
    std::string model_priority = "DEFAULT";         // High-level OpenVINO model priority hint
                                                    // Defines what model should be provided with more performant
                                                    // bounded resource first
    int num_streams = 1;                            // [num_streams]: Option that specifies the number of parallel
                                                    // inference requests to be processed on a given `device_type`.
                                                    // Overrides the accelerator default value of number of streams
                                                    // with this value at runtime.
    bool enable_opencl_throttling = false;          // [enable_opencl_throttling]: Enables OpenCL queue throttling for
                                                    // GPU device (Reduces CPU Utilization when using GPU)

    bool enable_qdq_optimizer = false;  // Enables QDQ pruning for efficient inference latency with NPU

    void* context = nullptr;

    std::string bool_flag = "";
    if (provider_options_map.find("device_type") != provider_options_map.end()) {
      device_type = provider_options_map.at("device_type").c_str();

      std::set<std::string> ov_supported_device_types = {"CPU", "GPU",
                                                         "GPU.0", "GPU.1", "NPU"};
      std::set<std::string> deprecated_device_types = {"CPU_FP32", "GPU_FP32",
                                                       "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                       "GPU.0_FP16", "GPU.1_FP16"};
      OVDevices devices;
      std::vector<std::string> available_devices = devices.get_ov_devices();

      for (auto& device : available_devices) {
        if (ov_supported_device_types.find(device) == ov_supported_device_types.end()) {
          ov_supported_device_types.emplace(device);
        }
      }
      if (deprecated_device_types.find(device_type) != deprecated_device_types.end()) {
        std::string deprecated_device = device_type;
        int delimit = device_type.find("_");
        device_type = deprecated_device.substr(0, delimit);
        precision = deprecated_device.substr(delimit + 1);
        LOGS_DEFAULT(WARNING) << "[OpenVINO] Selected 'device_type' " + deprecated_device + " is deprecated. \n"
                              << "Update the 'device_type' to specified types 'CPU', 'GPU', 'GPU.0', "
                              << "'GPU.1', 'NPU' or from"
                              << " HETERO/MULTI/AUTO options and set 'precision' separately. \n";
      }
      if (!((ov_supported_device_types.find(device_type) != ov_supported_device_types.end()) ||
            (device_type.find("HETERO:") == 0) ||
            (device_type.find("MULTI:") == 0) ||
            (device_type.find("AUTO:") == 0))) {
        ORT_THROW(
            "[ERROR] [OpenVINO] You have selected wrong configuration value for the key 'device_type'. "
            "Select from 'CPU', 'GPU', 'NPU', 'GPU.x' where x = 0,1,2 and so on or from"
            " HETERO/MULTI/AUTO options available. \n");
      }
    }
    if (provider_options_map.find("device_id") != provider_options_map.end()) {
      std::string dev_id = provider_options_map.at("device_id").c_str();
      LOGS_DEFAULT(WARNING) << "[OpenVINO] The options 'device_id' is deprecated. "
                            << "Upgrade to set deice_type and precision session options.\n";
      if (dev_id == "CPU" || dev_id == "GPU" || dev_id == "NPU") {
        device_type = std::move(dev_id);
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported device_id is selected. Select from available options.");
      }
    }
    if (provider_options_map.find("precision") != provider_options_map.end()) {
      precision = provider_options_map.at("precision").c_str();
    }
    if (device_type.find("GPU") != std::string::npos) {
      if (precision == "") {
        precision = "FP16";
      } else if (precision != "ACCURACY" && precision != "FP16" && precision != "FP32") {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. GPU only supports FP32 / FP16. \n");
      }
    } else if (device_type.find("NPU") != std::string::npos) {
      if (precision == "" || precision == "ACCURACY" || precision == "FP16") {
        precision = "FP16";
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. NPU only supported FP16. \n");
      }
    } else if (device_type.find("CPU") != std::string::npos) {
      if (precision == "" || precision == "ACCURACY" || precision == "FP32") {
        precision = "FP32";
      } else {
        ORT_THROW("[ERROR] [OpenVINO] Unsupported inference precision is selected. CPU only supports FP32 . \n");
      }
    }

    if (provider_options_map.find("cache_dir") != provider_options_map.end()) {
      cache_dir = provider_options_map.at("cache_dir");
    }

    if (provider_options_map.find("load_config") != provider_options_map.end()) {
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
            target_map[key] = inner_map;
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

      load_config = parse_config(provider_options_map.at("load_config"));
    }

    if (provider_options_map.find("context") != provider_options_map.end()) {
      std::string str = provider_options_map.at("context");
      uint64_t number = std::strtoull(str.c_str(), nullptr, 16);
      context = reinterpret_cast<void*>(number);
    }

    if (provider_options_map.find("num_of_threads") != provider_options_map.end()) {
      if (!std::all_of(provider_options_map.at("num_of_threads").begin(),
                       provider_options_map.at("num_of_threads").end(), ::isdigit)) {
        ORT_THROW("[ERROR] [OpenVINO-EP] Number of threads should be a number. \n");
      }
      num_of_threads = std::stoi(provider_options_map.at("num_of_threads"));
      if (num_of_threads <= 0) {
        num_of_threads = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_threads' should be in the positive range.\n "
                              << "Executing with num_threads=1";
      }
    }

    if (provider_options_map.find("model_priority") != provider_options_map.end()) {
      model_priority = provider_options_map.at("model_priority").c_str();
      std::vector<std::string> supported_priorities({"LOW", "MEDIUM", "HIGH", "DEFAULT"});
      if (std::find(supported_priorities.begin(), supported_priorities.end(),
                    model_priority) == supported_priorities.end()) {
        model_priority = "DEFAULT";
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'model_priority' "
                              << "is not one of LOW, MEDIUM, HIGH, DEFAULT. "
                              << "Executing with model_priorty=DEFAULT";
      }
    }

    if (provider_options_map.find("num_streams") != provider_options_map.end()) {
      num_streams = std::stoi(provider_options_map.at("num_streams"));
      if (num_streams <= 0) {
        num_streams = 1;
        LOGS_DEFAULT(WARNING) << "[OpenVINO-EP] The value for the key 'num_streams' should be in the range of 1-8.\n "
                              << "Executing with num_streams=1";
      }
    }
    if (provider_options_map.find("enable_opencl_throttling") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_opencl_throttling");
      if (bool_flag == "true" || bool_flag == "True")
        enable_opencl_throttling = true;
      else if (bool_flag == "false" || bool_flag == "False")
        enable_opencl_throttling = false;
      bool_flag = "";
    }

    if (provider_options_map.find("enable_qdq_optimizer") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("enable_qdq_optimizer");
      if (bool_flag == "true" || bool_flag == "True")
        enable_qdq_optimizer = true;
      else if (bool_flag == "false" || bool_flag == "False")
        enable_qdq_optimizer = false;
      else
        ORT_THROW("[ERROR] [OpenVINO-EP] enable_qdq_optimiser should be a boolean.\n");
      bool_flag = "";
    }

    // [disable_dynamic_shapes]:  Rewrite dynamic shaped models to static shape at runtime and execute.
    // Always true for NPU plugin.
    bool disable_dynamic_shapes = false;
    if (device_type.find("NPU") != std::string::npos) {
      disable_dynamic_shapes = true;
    }
    if (provider_options_map.find("disable_dynamic_shapes") != provider_options_map.end()) {
      bool_flag = provider_options_map.at("disable_dynamic_shapes");
      if (bool_flag == "true" || bool_flag == "True") {
        disable_dynamic_shapes = true;
      } else if (bool_flag == "false" || bool_flag == "False") {
        if (device_type.find("NPU") != std::string::npos) {
          disable_dynamic_shapes = true;
          LOGS_DEFAULT(INFO) << "[OpenVINO-EP] The value for the key 'disable_dynamic_shapes' will be set to "
                             << "TRUE for NPU backend.\n ";
        } else {
          disable_dynamic_shapes = false;
        }
      }
      bool_flag = "";
    }

    return std::make_shared<OpenVINOProviderFactory>(device_type,
                                                     precision,
                                                     num_of_threads,
                                                     load_config,
                                                     cache_dir,
                                                     model_priority,
                                                     num_streams,
                                                     context,
                                                     enable_opencl_throttling,
                                                     disable_dynamic_shapes,
                                                     enable_qdq_optimizer,
                                                     config_options);
  }

  void Initialize() override {
  }

  void Shutdown() override {
  }
} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
