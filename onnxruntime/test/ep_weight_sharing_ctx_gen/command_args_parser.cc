// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "command_args_parser.h"

#include <string.h>
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

#include "test_configuration.h"

namespace onnxruntime {
namespace qnnctxgen {

/*static*/ void CommandLineParser::ShowUsage() {
  printf(
      "ep_weight_sharing_ctx_gen [options...] model1_path,model2_path\n"
      "Example: ./ep_weight_sharing_ctx_gen -i \"soc_model|60 htp_graph_finalization_optimization_mode|3\" -C \"ep.context_node_name_prefix|_part1\" ./model1.onnx,./model2.onnx\n"
      "Options:\n"
      "\t-e [qnn|tensorrt|openvino|vitisai]: Specifies the compile based provider 'qnn','tensorrt','openvino', 'vitisai'. "
      "Default:'qnn'.\n"
      "\t-v: Show verbose information.\n"
      "\t-C: Specify session configuration entries as key-value pairs: -C \"<key1>|<value1> <key2>|<value2>\" \n"
      "\t    Refer to onnxruntime_session_options_config_keys.h for valid keys and values. \n"
      "\t    Force ep.context_enable to 1 and ep.context_embed_mode to 0. Change ep.context_file_path is not allowed."
      "\t    [Example] -C \"ep.context_node_name_prefix|_part1\" \n"
      "\t-i: Specify EP specific runtime options as key value pairs. Different runtime options available are: \n"
      "\t    [Usage]: -i '<key1>|<value1> <key2>|<value2>'\n"
      "\n"
      "\t    [QNN only] [backend_path]: QNN backend path. e.g '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. default to HTP backend\n"
      "\t    [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).\n"
      "\t    [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: '0', '1', '2', '3', default is '0'.\n"
      "\t    [QNN only] [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown). \n"
      "\t    [QNN only] [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. eg: '0', '68', '69', '73', '75'. Defaults to '0' (none). \n"
      "\t    [QNN only] [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision. \n"
      "\t    Otherwise, it will be fp32 precision. Works for float32 model for HTP backend. Defaults to '1' (with FP16 precision.). \n"
      "\t    [QNN only] [offload_graph_io_quantization]: Offload graph input quantization and graph output dequantization to another EP (typically CPU EP). \n"
      "\t    Defaults to '1' (QNN EP handles the graph I/O quantization and dequantization). \n"
      "\t    [QNN only] [enable_htp_spill_fill_buffer]: Enable HTP spill file buffer, used while generating QNN context binary."
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

/*static*/ bool CommandLineParser::ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  int ch;
  while ((ch = getopt(argc, argv, ORT_TSTR("e:o:u:i:C:vh"))) != -1) {
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

          if (key == "backend_path" || key == "vtcm_mb" || key == "soc_model" || key == "htp_arch") {
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
                     key == "enable_htp_spill_fill_buffer") {
            std::unordered_set<std::string> supported_options = {"0", "1"};
            if (supported_options.find(value) == supported_options.end()) {
              std::ostringstream str_stream;
              std::copy(supported_options.begin(), supported_options.end(),
                        std::ostream_iterator<std::string>(str_stream, ","));
              std::string str = str_stream.str();
              ORT_THROW("Wrong value for " + key + ". select from: " + str);
            }
          } else {
            ORT_THROW(R"(Wrong key type entered. Choose from options: ['backend_path', 'vtcm_mb', 'htp_performance_mode',
 'htp_graph_finalization_optimization_mode', 'soc_model', 'htp_arch', 'enable_htp_fp16_precision',
 'offload_graph_io_quantization', 'enable_htp_spill_fill_buffer'])");
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

  ParsePaths(argv[0], test_config.model_file_paths);

  return true;
}

}  // namespace qnnctxgen
}  // namespace onnxruntime
