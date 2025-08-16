// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "command_args_parser.h"
#include "utils/utils.h"
#include "utils/strings_helper.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include <core/graph/constants.h>
#include <core/optimizer/graph_transformer_level.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/flags/usage_config.h"
#include "absl/flags/reflection.h"

static const onnxruntime::test::TestConfig& DefaultTestConfig() {
  static onnxruntime::test::TestConfig default_config{};
  return default_config;
}

ABSL_FLAG(bool, A, DefaultTestConfig().enable_cpu_mem_arena, "Disables memory arena.");
ABSL_FLAG(bool, v, DefaultTestConfig().verbose_logging_required, "Shows verbose information.");
ABSL_FLAG(int, c, DefaultTestConfig().concurrent_session_runs, "Specifies the number of Session::Run() to invoke simultaneously for each model.");
ABSL_FLAG(int, j, DefaultTestConfig().p_models, "Specifies the number of models to run simultaneously.");
ABSL_FLAG(int, r, DefaultTestConfig().repeat_count, "Specifies the number of times to repeat.");
ABSL_FLAG(bool, I, DefaultTestConfig().inference_mode, "Uses inference mode. Saves the inference result and skips the output value comparison.");
ABSL_FLAG(bool, M, DefaultTestConfig().enable_mem_pattern, "Disables memory pattern.");
ABSL_FLAG(std::string, n, "", "Specifies a single test case to run.");
ABSL_FLAG(std::string, e, "cpu", "Specifies the provider 'cpu', 'cuda', 'dnnl', 'tensorrt', 'vsinpu', 'openvino', 'rocm', 'migraphx', 'acl', 'dml', 'armnn', 'xnnpack', 'webgpu', 'nnapi', 'qnn', 'snpe' or 'coreml'.");
ABSL_FLAG(std::string, t, "1e-5", "Specifies custom relative tolerance values for output value comparison.");
ABSL_FLAG(std::string, a, "1e-5", "Specifies custom absolute tolerance values for output value comparison.");
ABSL_FLAG(bool, x, false, "Uses parallel executor, default (without -x): sequential executor.");
ABSL_FLAG(bool, p, DefaultTestConfig().pause, "Pauses after launch, can attach debugger and continue.");
ABSL_FLAG(int, o, DefaultTestConfig().graph_optimization_level, "Specifies graph optimization level. Default is 99 (all). Valid values are 0 (disable), 1 (basic), 2 (extended), 3 (layout), 99 (all).");
ABSL_FLAG(int, d, DefaultTestConfig().device_id, "Specifies the device id for multi-device (e.g. GPU). The value should > 0.");
ABSL_FLAG(std::string, C, "",
          "Specifies session configuration entries as key-value pairs:\n -C \"<key1>|<value1> <key2>|<value2>\" \n"
          "Refer to onnxruntime_session_options_config_keys.h for valid keys and values. \n"
          "[Example] -C \"session.disable_cpu_ep_fallback|1 ep.context_enable|1\" \n");
ABSL_FLAG(std::string, i, "",
          "Specifies EP specific runtime options as key-value pairs.\n Different runtime options available are: \n"
          "  [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>'\n"
          "\n"
          "  [QNN only] [backend_type]: QNN backend type. E.g., 'cpu', 'htp'. Mutually exclusive with 'backend_path'.\n"
          "  [QNN only] [backend_path]: QNN backend path. E.g., '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. Mutually exclusive with 'backend_type'.\n"
          "  [QNN only] [profiling_level]: QNN profiling level, options:  'basic', 'detailed', default 'off'.\n"
          "  [QNN only] [profiling_file_path]: QNN profiling file path if ETW not enabled.\n"
          "  [QNN only] [rpc_control_latency]: QNN rpc control latency. default to 10.\n"
          "  [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).\n"
          "  [QNN only] [htp_performance_mode]: QNN performance mode, options: 'burst', 'balanced', 'default', 'high_performance', \n"
          "  'high_power_saver', 'low_balanced', 'extreme_power_saver', 'low_power_saver', 'power_saver', 'sustained_high_performance'. Default to 'default'. \n"
          "  [QNN only] [op_packages]: QNN UDO package, allowed format: \n"
          "  op_packages|<op_type>:<op_package_path>:<interface_symbol_name>[:<target>],<op_type2>:<op_package_path2>:<interface_symbol_name2>[:<target2>]. \n"
          "  [QNN only] [qnn_context_priority]: QNN context priority, options: 'low', 'normal', 'normal_high', 'high'. Default to 'normal'. \n"
          "  0 means dump the QNN context binary into separate bin file and set the path in the Onnx skeleton model.\n"
          "  [QNN only] [qnn_saver_path]: QNN Saver backend path. e.g '/folderpath/libQnnSaver.so'.\n"
          "  [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: \n"
          "  '0', '1', '2', '3', default is '0'.\n"
          "  [QNN only] [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown). \n"
          "  [QNN only] [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. \n"
          "  Options are '0', '68', '69', '73', '75'. Defaults to '0' (none). \n"
          "  [QNN only] [device_id]: The ID of the device to use when setting 'htp_arch'. Defaults to '0' (for single device). \n"
          "  [QNN only] [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision. \n"
          "  Otherwise, it will be fp32 precision. Works for float32 model for HTP backend. Defaults to '1' (with FP16 precision.). \n"
          "  [QNN only] [offload_graph_io_quantization]: Offload graph input quantization and graph output dequantization to another EP (typically CPU EP). \n"
          "  Defaults to '0' (QNN EP handles the graph I/O quantization and dequantization). \n"
          "  [Example] [For QNN EP] -e qnn -i \"profiling_level|detailed backend_type|cpu\" \n"
          "\n"
          "  [SNPE only] [runtime]: SNPE runtime, options: 'CPU', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n"
          "  [SNPE only] [priority]: execution priority, options: 'low', 'normal'. \n"
          "  [SNPE only] [buffer_type]: options: 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. default: ITENSOR'. \n"
          "  [SNPE only] [enable_init_cache]: enable SNPE init caching feature, set to 1 to enabled it. Disabled by default. \n"
          "  [Example] [For SNPE EP] -e snpe -i \"runtime|CPU priority|low\" \n");
ABSL_FLAG(bool, z, DefaultTestConfig().set_denormal_as_zero, "Sets denormal as zero.");
ABSL_FLAG(bool, b, DefaultTestConfig().disable_ep_context_embed_mode, "Disables EP context embed mode.");
ABSL_FLAG(bool, f, DefaultTestConfig().ep_context_enable, "Enables EP context cache generation.");
ABSL_FLAG(std::string, plugin_ep_libs, "",
          "Specifies a list of plugin execution provider (EP) registration names and their corresponding shared libraries to register.\n"
          "[Usage]: --plugin_ep_libs \"plugin_ep_name_1|plugin_ep_1.dll plugin_ep_name_2|plugin_ep_2.dll ... \"");
ABSL_FLAG(std::string, plugin_eps, "", "Specifies a semicolon-separated list of plugin execution providers (EPs) to use.");
ABSL_FLAG(std::string, plugin_ep_options, "",
          "Specifies provider options for each EP listed in --plugin_eps. Options (key-value pairs) for each EP are separated by space and EPs are separated by semicolons.\n"
          "[Usage]: --plugin_ep_options \"ep_1_option_1_key|ep_1_option_1_value ...;ep_2_option_1_key|ep_2_option_1_value ...;... \" or \n"
          "--plugin_ep_options \";ep_2_option_1_key|ep_2_option_1_value ...;... \" or \n"
          "--plugin_ep_options \"ep_1_option_1_key|ep_1_option_1_value ...;;ep_3_option_1_key|ep_3_option_1_value ...;... \"");
ABSL_FLAG(bool, list_ep_devices, false, "Prints all available device indices and their properties (including metadata). This option makes the program exit early without performing inference.\n");
ABSL_FLAG(std::string, select_ep_devices, "", "Specifies a semicolon-separated list of device indices to add to the session and run with.");
ABSL_FLAG(bool, h, false, "Print program usage.");

namespace onnxruntime {
namespace test {

std::string CustomUsageMessage() {
  std::ostringstream oss;
  oss << "onnx_test_runner [options...] <data_root>\n\n";
  oss << "Note: Options may be specified with either a single dash(-option) or a double dash(--option). Both forms are accepted and treated identically.\n\n";
  oss << "Options:";

  return oss.str();
}

bool CommandLineParser::ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  // Following callback is to make sure all the ABSL flags defined above will be showed up when running with "--help".
  // Note: By default abseil only wants flags in binary's main. It expects the main routine to reside in <program>.cc or <program>-main.cc or
  // <program>_main.cc, where the <program> is the name of the binary (without .exe on Windows). See usage_config.cc in abseil for more details.
  absl::FlagsUsageConfig config;
  config.contains_help_flags = [](absl::string_view filename) {
    return std::filesystem::path(filename).filename() == std::filesystem::path(__FILE__).filename();
  };

  config.normalize_filename = [](absl::string_view f) {
    return std::string(f);
  };
  absl::SetFlagsUsageConfig(config);
  absl::SetProgramUsageMessage(CustomUsageMessage());

  auto utf8_argv_strings = test::utils::ConvertArgvToUtf8Strings(argc, argv);
  auto utf8_argv = test::utils::CStringsFromStrings(utf8_argv_strings);
  auto positional = absl::ParseCommandLine(static_cast<int>(utf8_argv.size()), utf8_argv.data());

  // -A
  test_config.enable_cpu_mem_arena = absl::GetFlag(FLAGS_A);

  // -v
  test_config.verbose_logging_required = absl::GetFlag(FLAGS_v);

  // -c
  if (absl::GetFlag(FLAGS_c) <= 0) return false;
  test_config.concurrent_session_runs = absl::GetFlag(FLAGS_c);

  // -j
  if (absl::GetFlag(FLAGS_j) <= 0) return false;
  test_config.p_models = absl::GetFlag(FLAGS_j);

  // -r
  if (absl::GetFlag(FLAGS_r) <= 0) return false;
  test_config.repeat_count = absl::GetFlag(FLAGS_r);

  // -I
  test_config.inference_mode = absl::GetFlag(FLAGS_I);

  // -M
  test_config.enable_mem_pattern = absl::GetFlag(FLAGS_M);

  // -n
  {
    const auto& whitelisted_test_case = absl::GetFlag(FLAGS_n);
    if (!whitelisted_test_case.empty()) {
      // Abseil doesn't support the same option being provided multiple times - only the last occurrence is applied.
      // To preserve the previous usage of '-n', where users may specify it multiple times to provide whitelisted tests,
      // we need to manually parse argv.
      for (int i = 1; i < argc; ++i) {
        auto utf8_arg = utf8_argv_strings[i];
        if (utf8_arg == ("-n") || utf8_arg == ("--n")) {
          auto value_idx = i + 1;
          if (value_idx >= argc || utf8_argv_strings[value_idx][0] == '-') {
            std::cerr << utf8_arg << " should be followed by a key-value pair." << std::endl;
            return false;
          }

          // run only some whitelisted tests
          // TODO: parse name str to an array
          test_config.whitelisted_test_cases.emplace_back(ToPathString(utf8_argv_strings[value_idx]));
        }
      }
    }
  }

  // -e
  {
    auto const& ep = absl::GetFlag(FLAGS_e);
    if (!ep.empty()) {
      if (ep == "cpu") {
        // do nothing
      } else if (ep == "cuda") {
        test_config.enable_cuda = true;
      } else if (ep == "dnnl") {
        test_config.enable_dnnl = true;
      } else if (ep == "openvino") {
        test_config.enable_openvino = true;
      } else if (ep == "tensorrt") {
        test_config.enable_tensorrt = true;
      } else if (ep == "qnn") {
        test_config.enable_qnn = true;
      } else if (ep == "snpe") {
        test_config.enable_snpe = true;
      } else if (ep == "nnapi") {
        test_config.enable_nnapi = true;
      } else if (ep == "vsinpu") {
        test_config.enable_vsinpu = true;
      } else if (ep == "coreml") {
        test_config.enable_coreml = true;
      } else if (ep == "dml") {
        test_config.enable_dml = true;
      } else if (ep == "acl") {
        test_config.enable_acl = true;
      } else if (ep == "armnn") {
        test_config.enable_armnn = true;
      } else if (ep == "rocm") {
        test_config.enable_rocm = true;
      } else if (ep == "migraphx") {
        test_config.enable_migraphx = true;
      } else if (ep == "xnnpack") {
        test_config.enable_xnnpack = true;
      } else if (ep == "webgpu") {
        test_config.enable_webgpu = true;
      } else {
        return false;
      }
    }
  }

  // -t
  {
    const auto& override_relative_tolerance_value = absl::GetFlag(FLAGS_t);
    if (!override_relative_tolerance_value.empty()) {
      test_config.override_tolerance = true;
      test_config.rtol = OrtStrtod<char>(override_relative_tolerance_value.c_str(), nullptr);
    }
  }

  // -a
  {
    const auto& override_absolute_tolerance_value = absl::GetFlag(FLAGS_t);
    if (!override_absolute_tolerance_value.empty()) {
      test_config.override_tolerance = true;
      test_config.atol = OrtStrtod<char>(override_absolute_tolerance_value.c_str(), nullptr);
    }
  }

  // -x
  if (absl::GetFlag(FLAGS_x)) test_config.execution_mode = ExecutionMode::ORT_PARALLEL;

  // -p
  if (absl::GetFlag(FLAGS_p)) test_config.pause = true;

  // -o
  {
    const auto optimization_level = absl::GetFlag(FLAGS_o);
    if (optimization_level != test_config.graph_optimization_level) {
      switch (optimization_level) {
        case ORT_DISABLE_ALL:
          test_config.graph_optimization_level = ORT_DISABLE_ALL;
          break;
        case ORT_ENABLE_BASIC:
          test_config.graph_optimization_level = ORT_ENABLE_BASIC;
          break;
        case ORT_ENABLE_EXTENDED:
          test_config.graph_optimization_level = ORT_ENABLE_EXTENDED;
          break;
        case ORT_ENABLE_LAYOUT:
          test_config.graph_optimization_level = ORT_ENABLE_LAYOUT;
          break;
        case ORT_ENABLE_ALL:
          test_config.graph_optimization_level = ORT_ENABLE_ALL;
          break;
        default: {
          if (optimization_level > ORT_ENABLE_ALL) {  // relax constraint
            test_config.graph_optimization_level = ORT_ENABLE_ALL;
          } else {
            return false;
          }
        }
      }
      test_config.user_graph_optimization_level_set = true;
    }
  }

  // -d
  test_config.device_id = absl::GetFlag(FLAGS_d);

  // -C
  {
    const auto& session_configs = absl::GetFlag(FLAGS_C);
    if (!session_configs.empty()) {
      ORT_TRY {
        test::utils::ParseSessionConfigs(session_configs, test_config.session_config_entries);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          fprintf(stderr, "Error parsing session configuration entries: %s\n", ex.what());
        });
        return false;
      }
    }
  }

  // -i
  {
    const auto& ep_options = absl::GetFlag(FLAGS_i);
    if (!ep_options.empty()) test_config.ep_runtime_config_string = ToPathString(ep_options);
  }

  // -z
  if (absl::GetFlag(FLAGS_z)) test_config.set_denormal_as_zero = true;

  // -b
  if (absl::GetFlag(FLAGS_b)) test_config.disable_ep_context_embed_mode = true;

  // -f
  if (absl::GetFlag(FLAGS_f)) test_config.ep_context_enable = true;

  // --plugin_ep_libs
  {
    const auto& plugin_ep_names_and_libs = absl::GetFlag(FLAGS_plugin_ep_libs);
    if (!plugin_ep_names_and_libs.empty()) test_config.plugin_ep_names_and_libs = ToPathString(plugin_ep_names_and_libs);
  }

  // --plugin_eps
  {
    const auto& plugin_eps = absl::GetFlag(FLAGS_plugin_eps);
    if (!plugin_eps.empty()) test::utils::ParseEpList(plugin_eps, test_config.plugin_ep_list);
  }

  // --plugin_ep_options
  {
    const auto& plugin_ep_options = absl::GetFlag(FLAGS_plugin_ep_options);
    if (!plugin_ep_options.empty()) test_config.ep_runtime_config_string = ToPathString(plugin_ep_options);
  }

  // --list_ep_devices
  if (absl::GetFlag(FLAGS_list_ep_devices)) {
    test_config.list_available_ep_devices = true;
    return true;
  }

  // --select_ep_devices
  {
    const auto& select_ep_devices = absl::GetFlag(FLAGS_select_ep_devices);
    if (!select_ep_devices.empty()) test_config.selected_ep_device_indices = select_ep_devices;
  }

  if (positional.size() > 1) {
    for (size_t i = 1; i < positional.size(); ++i) {
      test_config.data_dirs.emplace_back(ToPathString(positional[i]));
    }
  } else {
    std::cerr << "Please specify a test data dir" << std::endl;
    return false;
  }

  return true;
}

}  // namespace test
}  // namespace onnxruntime
