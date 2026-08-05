// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "command_args_parser.h"

#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unordered_map>

// Windows Specific
#ifdef _WIN32
#include "getopt.h"
#include "windows.h"
#else
#include <unistd.h>
#endif

#include <core/graph/constants.h>
#include <core/platform/path_lib.h>
#include <core/optimizer/graph_transformer_level.h>

#include "nlohmann/json.hpp"
#include "test_configuration.h"

namespace onnxruntime {
namespace qnnctxgen {

/*static*/ void CommandLineParser::ShowUsage() {
  printf(
      "%s",
      "ep_weight_sharing_ctx_gen [options...] model1_path,model2_path\n"
      "\n"
      "Example: ./ep_weight_sharing_ctx_gen -e qnn -i \"soc_model|60 htp_graph_finalization_optimization_mode|3\" -C \"ep.context_node_name_prefix|_part1\" ./model1.onnx,./model2.onnx\n"
      "\n"
      "Options:\n"
      "\t-e [qnn|tensorrt|openvino|vitisai]: Specifies the compile based provider 'qnn', 'tensorrt', 'openvino', 'vitisai'. Default: 'qnn'.\n"
      "\t-p [plugin_ep_config_json_file]: Specify JSON configuration file for a plugin EP. Takes precedence over the '-e' and '-i' options.\n"
      "\n"
      "\t                                 Example JSON configuration that selects plugin EP devices via EP name:\n"
      "\t                                   {\n"
      "\t                                     \"ep_library_registration_name\": \"example_plugin_ep\",\n"
      "\t                                     \"ep_library_path\": \"example_plugin_ep.dll\",\n"
      "\t                                     \"selected_ep_name\": \"example_plugin_ep\",\n"
      "\t                                     \"default_ep_options\": { \"key\": \"value\" }\n"
      "\t                                   }\n"
      "\n"
      "\t                                 Example JSON configuration that selects plugin EP devices via index:\n"
      "\t                                   {\n"
      "\t                                     \"ep_library_registration_name\": \"example_plugin_ep\",\n"
      "\t                                     \"ep_library_path\": \"example_plugin_ep.dll\",\n"
      "\t                                     \"selected_ep_device_indices\": [ 0 ],\n"
      "\t                                     \"default_ep_options\": { \"key\": \"value\" }\n"
      "\t                                   }\n"
      "\t-v: Show verbose information.\n"
      "\t-C: Specify session configuration entries as key-value pairs: -C \"<key1>|<value1> <key2>|<value2>\" \n"
      "\t    Refer to onnxruntime_session_options_config_keys.h for valid keys and values. \n"
      "\t    Force ep.context_enable to 1 and ep.context_embed_mode to 0. Change ep.context_file_path is not allowed.\n"
      "\t    [Example] -C \"ep.context_node_name_prefix|_part1\" \n"
      "\t-i: Specify EP specific runtime options as key value pairs. Different runtime options available are: \n"
      "\t    [Usage]: -i '<key1>|<value1> <key2>|<value2>'\n"
      "\n"
      "\t    [QNN only] [backend_type]: QNN backend type. E.g., 'cpu', 'htp'. Mutually exclusive with 'backend_path'.\n"
      "\t    [QNN only] [backend_path]: QNN backend path. E.g., '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. Mutually exclusive with 'backend_type'.\n"
      "\t    [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).\n"
      "\t    [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: '0', '1', '2', '3', default is '0'.\n"
      "\t    [QNN only] [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown). \n"
      "\t    [QNN only] [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. eg: '0', '68', '69', '73', '75'. Defaults to '0' (none). \n"
      "\t    [QNN only] [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision. \n"
      "\t    Otherwise, it will be fp32 precision. Works for float32 model for HTP backend. Defaults to '1' (with FP16 precision.). \n"
      "\t    [QNN only] [offload_graph_io_quantization]: Offload graph input quantization and graph output dequantization to another EP (typically CPU EP). \n"
      "\t    Defaults to '1' (another EP (typically CPU EP) handles the graph I/O quantization and dequantization). \n"
      "\t    [QNN only] [enable_htp_spill_fill_buffer]: Enable HTP spill file buffer, used while generating QNN context binary.\n"
      "\t    [QNN only] [extended_udma]: Enable HTP extended UDMA mode for better performance on supported hardware, options: \n"
      "\t    '0' (disabled), '1' (enabled). Default: '0'. \n"
      "\t    [Example] -i \"vtcm_mb|8 htp_arch|73\" \n"
      "\n"
      "\t-h: help\n");
}

#ifdef _WIN32
static const ORTCHAR_T* delimiter = L",";
#else
static const ORTCHAR_T* delimiter = ",";
#endif
static void ParsePaths(const std::basic_string<ORTCHAR_T>& path, std::vector<std::basic_string<ORTCHAR_T>>& paths) {
  std::basic_string<ORTCHAR_T> path_str(path);
  size_t pos = 0;
  std::basic_string<ORTCHAR_T> token;
  while (pos = path_str.find(delimiter), pos != std::string::npos) {
    token = path_str.substr(0, pos);
    paths.push_back(token);
    path_str.erase(0, pos + 1);
  }
  paths.push_back(path_str);

  return;
}

static bool ParseSessionConfigs(const std::string& configs_string,
                                std::unordered_map<std::string, std::string>& session_configs) {
  std::istringstream ss(configs_string);
  std::string token;

  while (ss >> token) {
    if (token == "") {
      continue;
    }

    std::string_view token_sv(token);

    auto pos = token_sv.find("|");
    if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length()) {
      // Error: must use a '|' to separate the key and value for session configuration entries.
      return false;
    }

    std::string key(token_sv.substr(0, pos));
    std::string value(token_sv.substr(pos + 1));

    auto it = session_configs.find(key);
    if (it != session_configs.end()) {
      // Error: specified duplicate session configuration entry: {key}
      return false;
    }

    session_configs.insert(std::make_pair(std::move(key), std::move(value)));
  }

  return true;
}

static bool ParsePluginEpConfig(const std::string& json_file_path, PluginEpConfig& config_out) {
  using json = nlohmann::json;
  bool success = true;

  ORT_TRY {
    std::ifstream ifs{json_file_path};
    if (!ifs) {
      std::cerr << "ERROR: Failed to open plugin EP configuration file at path: "
                << json_file_path.c_str() << std::endl;
      return false;
    }

    std::string content(std::istreambuf_iterator<char>{ifs},
                        std::istreambuf_iterator<char>{});
    PluginEpConfig config{};
    const auto parsed_json = json::parse(content);

    // required keys
    parsed_json.at("ep_library_registration_name").get_to(config.ep_library_registration_name);
    parsed_json.at("ep_library_path").get_to(config.ep_library_path);

    // optional keys
    config.default_ep_options = parsed_json.value<decltype(config.default_ep_options)>("default_ep_options", {});
    config.selected_ep_name = parsed_json.value<decltype(config.selected_ep_name)>("selected_ep_name", {});
    config.selected_ep_device_indices =
        parsed_json.value<decltype(config.selected_ep_device_indices)>("selected_ep_device_indices", {});

    if (config.selected_ep_name.empty() == config.selected_ep_device_indices.empty()) {
      std::cerr << "ERROR: Plugin EP configuration must specify exactly one of 'selected_ep_name' "
                << "or 'selected_ep_device_indices'" << std::endl;
      return false;
    }

    config_out = std::move(config);
    return success;
  }
  ORT_CATCH(const json::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::string kExampleValidJsonStr =
          "{\n"
          "  \"ep_library_registration_name\": \"example_plugin_ep\",\n"
          "  \"ep_library_path\": \"/path/to/example_plugin_ep.dll\",\n"
          "  \"selected_ep_name\": \"example_plugin_ep\"\n"
          "}";

      success = false;
      std::cerr << "ERROR: JSON parse error: " << e.what() << std::endl;
      std::cerr << "This is an example valid JSON configuration:\n"
                << kExampleValidJsonStr.c_str() << std::endl;
    });
  }
  return success;
}

/*static*/ bool CommandLineParser::ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  int ch;
  while ((ch = getopt(argc, argv, ORT_TSTR("e:p:o:u:i:C:vh"))) != -1) {
    switch (ch) {
      case 'e':
        if (!CompareCString(optarg, ORT_TSTR("qnn"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kQnnExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kOpenVINOExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kTensorrtExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("vitisai"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kVitisAIExecutionProvider;
        } else {
          fprintf(stderr, "The execution provider is not included in this tool.\n");
          return false;
        }
        break;
      case 'p': {
#ifdef _MSC_VER
        std::string plugin_ep_config_file_path = ToUTF8String(optarg);
#else
        std::string plugin_ep_config_file_path = optarg;
#endif
        PluginEpConfig plugin_ep_config{};
        if (!ParsePluginEpConfig(plugin_ep_config_file_path, plugin_ep_config)) {
          return false;
        }

        test_config.machine_config.plugin_ep_config = std::move(plugin_ep_config);
        break;
      }
      case 'v':
        test_config.run_config.f_verbose = true;
        break;
      case 'i': {
#ifdef _MSC_VER
        std::string option_string = ToUTF8String(optarg);
#else
        std::string option_string = optarg;
#endif
        std::istringstream ss(option_string);
        std::string token;

        while (ss >> token) {
          if (token == "") {
            continue;
          }
          auto pos = token.find("|");
          if (pos == std::string::npos || pos == 0 || pos == token.length()) {
            ORT_THROW("Use a '|' to separate the key and value for the run-time option you are trying to use.");
          }

          std::string key(token.substr(0, pos));
          std::string value(token.substr(pos + 1));

          if (key == "backend_type" || key == "backend_path" || key == "vtcm_mb" || key == "soc_model" ||
              key == "htp_arch") {
            // no validation
          } else if (key == "htp_graph_finalization_optimization_mode") {
            std::unordered_set<std::string> supported_htp_graph_final_opt_modes = {"0", "1", "2", "3"};
            if (supported_htp_graph_final_opt_modes.find(value) == supported_htp_graph_final_opt_modes.end()) {
              std::ostringstream str_stream;
              std::copy(supported_htp_graph_final_opt_modes.begin(), supported_htp_graph_final_opt_modes.end(),
                        std::ostream_iterator<std::string>(str_stream, ","));
              std::string str = str_stream.str();
              ORT_THROW("Wrong value for htp_graph_finalization_optimization_mode. select from: " + str);
            }
          } else if (key == "enable_htp_fp16_precision" || key == "offload_graph_io_quantization" ||
                     key == "enable_htp_spill_fill_buffer" || key == "extended_udma") {
            std::unordered_set<std::string> supported_options = {"0", "1"};
            if (supported_options.find(value) == supported_options.end()) {
              std::ostringstream str_stream;
              std::copy(supported_options.begin(), supported_options.end(),
                        std::ostream_iterator<std::string>(str_stream, ","));
              std::string str = str_stream.str();
              ORT_THROW("Wrong value for " + key + ". select from: " + str);
            }
          } else {
            ORT_THROW(
                "Wrong key type entered. Choose from options: ['backend_type', 'backend_path', 'vtcm_mb', "
                "'htp_performance_mode', 'htp_graph_finalization_optimization_mode', 'soc_model', 'htp_arch', "
                "'enable_htp_fp16_precision', 'offload_graph_io_quantization', 'enable_htp_spill_fill_buffer', 'extended_udma']");
          }

          test_config.run_config.provider_options[key] = value;
        }
        break;
      }
      case 'C': {
        if (!ParseSessionConfigs(ToUTF8String(optarg), test_config.run_config.session_config_entries)) {
          return false;
        }
        break;
      }
      case '?':
      case 'h':
      default:
        return false;
    }
  }

  // parse model_path
  argc -= optind;
  argv += optind;

  if (argc == 0) {
    std::cerr << "ERROR: Did not specify model paths" << std::endl;
    return false;
  }

  ParsePaths(argv[0], test_config.model_file_paths);

  return true;
}

}  // namespace qnnctxgen
}  // namespace onnxruntime
