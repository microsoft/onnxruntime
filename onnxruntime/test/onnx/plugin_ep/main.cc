// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
// For memory leak detection on Windows with Visual Studio in Debug mode
#ifdef _WIN32
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "TestResultStat.h"
#include "TestCase.h"
#include "testenv.h"
#include "test/util/include/providers.h"

#include "command_args_parser.h"
#include "utils/utils.h"
#include "utils/strings_helper.h"

#include <google/protobuf/stubs/common.h>
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif

using namespace onnxruntime;

namespace {
static TestTolerances LoadTestTolerances(bool enable_cuda, bool enable_openvino, bool useCustom, double atol, double rtol) {
  TestTolerances::Map absolute_overrides;
  TestTolerances::Map relative_overrides;
  if (useCustom) {
    return TestTolerances(atol, rtol, absolute_overrides, relative_overrides);
  }
  std::ifstream overrides_ifstream(ConcatPathComponent(
      ORT_TSTR("testdata"), ORT_TSTR("onnx_backend_test_series_overrides.jsonc")));
  if (!overrides_ifstream.good()) {
    constexpr double absolute = 1e-3;
    // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
    // when openvino is enabled, set it to a larger value for resolving MNIST accuracy mismatch
    const double relative = enable_cuda ? 0.017 : enable_openvino ? 0.009
                                                                  : 1e-3;
    return TestTolerances(absolute, relative, absolute_overrides, relative_overrides);
  }
  const auto overrides_json = nlohmann::json::parse(
      overrides_ifstream,
      /*cb=*/nullptr, /*allow_exceptions=*/true
// Comment support is added in 3.9.0 with breaking change to default behavior.
#if NLOHMANN_JSON_VERSION_MAJOR * 1000 + NLOHMANN_JSON_VERSION_MINOR >= 3009
      ,
      /*ignore_comments=*/true
#endif
  );
  overrides_json["atol_overrides"].get_to(absolute_overrides);
  overrides_json["rtol_overrides"].get_to(relative_overrides);
  return TestTolerances(
      overrides_json["atol_default"], overrides_json["rtol_default"], absolute_overrides, relative_overrides);
}
}  // namespace

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[], Ort::Env& env) {
#else
int real_main(int argc, char* argv[], Ort::Env& env) {
#endif
  test::TestConfig test_config;
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_ERROR;

  if (!test::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    std::cerr << "See 'onnx_test_runner --help'." << std::endl;
    return -1;
  }

  // TODO: Support specifying all valid levels of logging
  // Currently the logging level is ORT_LOGGING_LEVEL_ERROR by default and
  // if the user adds -v, the logging level is ORT_LOGGING_LEVEL_VERBOSE
  if (test_config.verbose_logging_required) {
    logging_level = ORT_LOGGING_LEVEL_VERBOSE;
  }

  if (test_config.pause) {
    printf("Enter to continue...\n");
    fflush(stdout);
    (void)getchar();
  }

  {
    bool failed = false;
    ORT_TRY {
      env = Ort::Env{logging_level, "Default"};
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        fprintf(stderr, "Error creating environment: %s \n", ex.what());
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  if (!test_config.plugin_ep_names_and_libs.empty()) {
    test::utils::RegisterExecutionProviderLibrary(env, test_config.plugin_ep_names_and_libs, test_config.registered_plugin_eps);
  }

  auto unregister_plugin_eps_at_scope_exit = gsl::finally([&]() {
    if (!test_config.registered_plugin_eps.empty()) {
      test::utils::UnregisterExecutionProviderLibrary(env, test_config.registered_plugin_eps);
    }
  });

  TestResultStat stat;

  std::vector<std::unique_ptr<ITestCase>> owned_tests;
  {
    Ort::SessionOptions sf;

    // Add EP devices if any (created by plugin EP)
    if (!test_config.registered_plugin_eps.empty()) {
      std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();
      // EP -> associated EP devices (All OrtEpDevice instances must be from the same execution provider)
      std::unordered_map<std::string, std::vector<Ort::ConstEpDevice>> added_ep_devices;
      std::unordered_set<int> added_ep_device_index_set;

      auto& ep_list = test_config.plugin_ep_list;
      std::unordered_set<std::string> ep_set(ep_list.begin(), ep_list.end());

      // Select EP devices by provided device index
      if (!test_config.selected_ep_device_indices.empty()) {
        std::vector<int> device_list;
        device_list.reserve(test_config.selected_ep_device_indices.size());
        test::utils::ParseEpDeviceIndexList(test_config.selected_ep_device_indices, device_list);
        for (auto index : device_list) {
          if (static_cast<size_t>(index) > (ep_devices.size() - 1)) {
            fprintf(stderr, "%s", "The device index provided is not correct. Will skip this device id.");
            continue;
          }

          Ort::ConstEpDevice& device = ep_devices[index];
          if (ep_set.find(std::string(device.EpName())) != ep_set.end()) {
            if (added_ep_device_index_set.find(index) == added_ep_device_index_set.end()) {
              added_ep_devices[device.EpName()].push_back(device);
              added_ep_device_index_set.insert(index);
              fprintf(stdout, "[Plugin EP] EP Device [Index: %d, Name: %s] has been added to session.\n", index, device.EpName());
            }
          } else {
            std::string ep_list_string;
            for (size_t i = 0; i < ep_list.size(); ++i) {
              ep_list_string += ep_list[i];
              if (i + 1 < ep_list.size()) {
                ep_list_string += ", ";
              }
            }
            std::string err_msg = "[Plugin EP] [WARNING] : The EP device index and its corresponding OrtEpDevice is not created from " +
                                  ep_list_string + ". Will skip adding this device.\n";
            fprintf(stderr, "%s", err_msg.c_str());
          }
        }
      } else {
        // Find and select the OrtEpDevice associated with the EP in "--plugin_eps".
        for (size_t index = 0; index < ep_devices.size(); ++index) {
          Ort::ConstEpDevice& device = ep_devices[index];
          if (ep_set.find(std::string(device.EpName())) != ep_set.end()) {
            added_ep_devices[device.EpName()].push_back(device);
            fprintf(stdout, "EP Device [Index: %d, Name: %s] has been added to session.\n", static_cast<int>(index), device.EpName());
          }
        }
      }

      if (added_ep_devices.empty()) {
        ORT_THROW("[ERROR] [Plugin EP]: No matching EP devices found.");
      }

      std::string ep_option_string = ToUTF8String(test_config.ep_runtime_config_string);

      // EP's associated provider option lists
      std::vector<std::unordered_map<std::string, std::string>> ep_options_list;
      test::utils::ParseEpOptions(ep_option_string, ep_options_list);

      // If user only provide the EPs' provider option lists for the first several EPs,
      // add empty provider option lists for the rest EPs.
      if (ep_options_list.size() < ep_list.size()) {
        for (size_t i = ep_options_list.size(); i < ep_list.size(); ++i) {
          ep_options_list.emplace_back();  // Adds a new empty map
        }
      } else if (ep_options_list.size() > ep_list.size()) {
        ORT_THROW("[ERROR] [Plugin EP]: Too many EP provider option lists provided.");
      }

      // EP -> associated provider options
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>> ep_options_map;
      for (size_t i = 0; i < ep_list.size(); ++i) {
        ep_options_map.emplace(ep_list[i], ep_options_list[i]);
      }

      for (auto& ep_and_devices : added_ep_devices) {
        auto& ep = ep_and_devices.first;
        auto& devices = ep_and_devices.second;
        sf.AppendExecutionProvider_V2(env, devices, ep_options_map[ep]);
      }
    }

    if (test_config.enable_cpu_mem_arena)
      sf.EnableCpuMemArena();
    else
      sf.DisableCpuMemArena();
    if (test_config.enable_mem_pattern)
      sf.EnableMemPattern();
    else
      sf.DisableMemPattern();
    sf.SetExecutionMode(test_config.execution_mode);
    if (test_config.set_denormal_as_zero)
      sf.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

    if (test_config.ep_context_enable)
      sf.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    if (test_config.disable_ep_context_embed_mode) {
      sf.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
    } else {
      sf.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    }

    for (auto& it : test_config.session_config_entries) {
      sf.AddConfigEntry(it.first.c_str(), it.second.c_str());
    }

    if (test_config.enable_tensorrt) {
#ifdef USE_TENSORRT
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, test_config.device_id));
#ifdef USE_CUDA
      OrtCUDAProviderOptionsV2 cuda_options;
      cuda_options.device_id = test_config.device_id;
      cuda_options.do_copy_in_default_stream = true;
      cuda_options.use_tf32 = false;
      // TODO: Support arena configuration for users of test runner
      sf.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif
#else
      fprintf(stderr, "TensorRT is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_openvino) {
#ifdef USE_OPENVINO
      // Setting default optimization level for OpenVINO can be overridden with -o option
      sf.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      sf.AppendExecutionProvider_OpenVINO(OrtOpenVINOProviderOptions{});
#else
      fprintf(stderr, "OpenVINO is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_cuda) {
#ifdef USE_CUDA
      OrtCUDAProviderOptionsV2 cuda_options;
      cuda_options.device_id = test_config.device_id;
      cuda_options.do_copy_in_default_stream = true;
      cuda_options.use_tf32 = false;
      // TODO: Support arena configuration for users of test runner
      sf.AppendExecutionProvider_CUDA_V2(cuda_options);
#else
      fprintf(stderr, "CUDA is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_dnnl) {
#ifdef USE_DNNL
      // Generate dnnl_options to optimize dnnl performance
      OrtDnnlProviderOptions dnnl_options;
      dnnl_options.use_arena = enable_cpu_mem_arena ? 1 : 0;
      dnnl_options.threadpool_args = nullptr;
#if defined(DNNL_ORT_THREAD)
      dnnl_options.threadpool_args = static_cast<void*>(TestEnv::GetDefaultThreadPool(Env::Default()));
#endif  // defined(DNNL_ORT_THREAD)
      sf.AppendExecutionProvider_Dnnl(dnnl_options);
#else
      fprintf(stderr, "DNNL is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_qnn) {
#ifdef USE_QNN
#ifdef _MSC_VER
      std::string option_string = ToUTF8String(test_config.ep_runtime_config_string);
#else
      std::string option_string = test_config.ep_runtime_config_string;
#endif
      std::istringstream ss(option_string);
      std::string token;
      std::unordered_map<std::string, std::string> qnn_options;

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

        if (key == "backend_path" || key == "profiling_file_path") {
          if (value.empty()) {
            ORT_THROW("Please provide the valid file path.");
          }
        } else if (key == "qnn_context_embed_mode") {
          if (value != "0") {
            ORT_THROW("Set to 0 to disable qnn_context_embed_mode.");
          }
        } else if (key == "profiling_level") {
          std::set<std::string> supported_profiling_level = {"off", "basic", "detailed"};
          if (supported_profiling_level.find(value) == supported_profiling_level.end()) {
            ORT_THROW("Supported profiling_level: off, basic, detailed");
          }
        } else if (key == "backend_type" || key == "rpc_control_latency" || key == "vtcm_mb" || key == "soc_model" ||
                   key == "device_id") {
          // no validation
        } else if (key == "htp_performance_mode") {
          std::set<std::string> supported_htp_perf_mode = {"burst", "balanced", "default", "high_performance",
                                                           "high_power_saver", "low_balanced", "extreme_power_saver", "low_power_saver",
                                                           "power_saver", "sustained_high_performance"};
          if (supported_htp_perf_mode.find(value) == supported_htp_perf_mode.end()) {
            std::ostringstream str_stream;
            std::copy(supported_htp_perf_mode.begin(), supported_htp_perf_mode.end(),
                      std::ostream_iterator<std::string>(str_stream, ","));
            std::string str = str_stream.str();
            ORT_THROW("Wrong value for htp_performance_mode. select from: " + str);
          }
        } else if (key == "op_packages") {
          if (value.empty()) {
            ORT_THROW("Please provide the valid op_packages.");
          }
        } else if (key == "qnn_context_priority") {
          std::set<std::string> supported_qnn_context_priority = {"low", "normal", "normal_high", "high"};
          if (supported_qnn_context_priority.find(value) == supported_qnn_context_priority.end()) {
            ORT_THROW("Supported qnn_context_priority: low, normal, normal_high, high");
          }
        } else if (key == "qnn_saver_path") {
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
        } else if (key == "htp_arch") {
          std::unordered_set<std::string> supported_htp_archs = {"0", "68", "69", "73", "75"};
          if (supported_htp_archs.find(value) == supported_htp_archs.end()) {
            std::ostringstream str_stream;
            std::copy(supported_htp_archs.begin(), supported_htp_archs.end(),
                      std::ostream_iterator<std::string>(str_stream, ","));
            std::string str = str_stream.str();
            ORT_THROW("Wrong value for htp_arch. select from: " + str);
          }
        } else if (key == "enable_htp_fp16_precision" || key == "offload_graph_io_quantization") {
          std::unordered_set<std::string> supported_options = {"0", "1"};
          if (supported_options.find(value) == supported_options.end()) {
            std::ostringstream str_stream;
            std::copy(supported_options.begin(), supported_options.end(),
                      std::ostream_iterator<std::string>(str_stream, ","));
            std::string str = str_stream.str();
            ORT_THROW("Wrong value for ", key, ". select from: ", str);
          }
        } else {
          ORT_THROW(
              "Wrong key type entered. Choose from options: ['backend_type', 'backend_path', "
              "'profiling_level', 'profiling_file_path', 'rpc_control_latency', 'vtcm_mb', 'htp_performance_mode', "
              "'qnn_saver_path', 'htp_graph_finalization_optimization_mode', 'op_packages', 'qnn_context_priority', "
              "'soc_model', 'htp_arch', 'device_id', 'enable_htp_fp16_precision', 'offload_graph_io_quantization']");
        }

        qnn_options[key] = value;
      }
      sf.AppendExecutionProvider("QNN", qnn_options);
#else
      fprintf(stderr, "QNN is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_nnapi) {
#ifdef USE_NNAPI
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf, 0));
#else
      fprintf(stderr, "NNAPI is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_vsinpu) {
#ifdef USE_VSINPU
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_VSINPU(sf));
#else
      fprintf(stderr, "VSINPU is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_coreml) {
#ifdef USE_COREML
      sf.AppendExecutionProvider("CoreML", {});
#else
      fprintf(stderr, "CoreML is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_snpe) {
#ifdef USE_SNPE
#ifdef _MSC_VER
      std::string option_string = ToUTF8String(test_config.ep_runtime_config_string);
#else
      std::string option_string = test_config.ep_runtime_config_string;
#endif
      std::istringstream ss(option_string);
      std::string token;
      std::unordered_map<std::string, std::string> snpe_options;

      while (ss >> token) {
        if (token == "") {
          continue;
        }
        auto pos = token.find("|");
        if (pos == std::string::npos || pos == 0 || pos == token.length()) {
          ORT_THROW(R"(Use a '|' to separate the key and value for
the run-time option you are trying to use.\n)");
        }

        std::string key(token.substr(0, pos));
        std::string value(token.substr(pos + 1));

        if (key == "runtime") {
          std::set<std::string> supported_runtime = {"CPU", "GPU_FP32", "GPU", "GPU_FLOAT16", "DSP", "AIP_FIXED_TF"};
          if (supported_runtime.find(value) == supported_runtime.end()) {
            ORT_THROW(R"(Wrong configuration value for the key 'runtime'.
select from 'CPU', 'GPU_FP32', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n)");
          }
        } else if (key == "priority") {
          // no validation
        } else if (key == "buffer_type") {
          std::set<std::string> supported_buffer_type = {"TF8", "TF16", "UINT8", "FLOAT", "ITENSOR"};
          if (supported_buffer_type.find(value) == supported_buffer_type.end()) {
            ORT_THROW(R"(Wrong configuration value for the key 'buffer_type'.
select from 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. \n)");
          }
        } else if (key == "enable_init_cache") {
          if (value != "1") {
            ORT_THROW("Set to 1 to enable_init_cache.");
          }
        } else {
          ORT_THROW("Wrong key type entered. Choose from options: ['runtime', 'priority', 'buffer_type', 'enable_init_cache'] \n");
        }

        snpe_options[key] = value;
      }

      sf.AppendExecutionProvider("SNPE", snpe_options);
#else
      fprintf(stderr, "SNPE is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_dml) {
#ifdef USE_DML
      fprintf(stderr, "Disabling mem pattern and forcing single-threaded execution since DML is used");
      sf.DisableMemPattern();
      sf.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      p_models = 1;
      concurrent_session_runs = 1;
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sf, device_id));
#else
      fprintf(stderr, "DML is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_acl) {
#ifdef USE_ACL
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(sf, false));
#else
      fprintf(stderr, "ACL is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_armnn) {
#ifdef USE_ARMNN
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "ArmNN is not supported in this build\n");
      return -1;
#endif
    }
    if (test_config.enable_rocm) {
#ifdef USE_ROCM
      OrtROCMProviderOptions rocm_options;
      rocm_options.do_copy_in_default_stream = true;
      // TODO: Support arena configuration for users of test runner
      sf.AppendExecutionProvider_ROCM(rocm_options);
#else
      fprintf(stderr, "ROCM is not supported in this build");
      return -1;
#endif
    }
    if (test_config.enable_migraphx) {
#ifdef USE_MIGRAPHX
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sf, device_id));
#else
      fprintf(stderr, "MIGRAPHX is not supported in this build");
      return -1;
#endif
    }

    if (test_config.enable_xnnpack) {
#ifdef USE_XNNPACK
      sf.AppendExecutionProvider("XNNPACK", {});
#else
      fprintf(stderr, "XNNPACK is not supported in this build");
      return -1;
#endif
    }

    if (test_config.enable_webgpu) {
#ifdef USE_WEBGPU
      sf.AppendExecutionProvider("WebGPU", {});
#else
      fprintf(stderr, "WebGPU is not supported in this build");
      return -1;
#endif
    }

    if (test_config.user_graph_optimization_level_set) {
      sf.SetGraphOptimizationLevel(test_config.graph_optimization_level);
    }

    // TODO: Get these from onnx_backend_test_series_filters.jsonc.
    // Permanently exclude following tests because ORT support only opset staring from 7,
    // Please make no more changes to the list
    static const ORTCHAR_T* immutable_broken_tests[] =
        {
            ORT_TSTR("AvgPool1d"),
            ORT_TSTR("AvgPool1d_stride"),
            ORT_TSTR("AvgPool2d"),
            ORT_TSTR("AvgPool2d_stride"),
            ORT_TSTR("AvgPool3d"),
            ORT_TSTR("AvgPool3d_stride"),
            ORT_TSTR("AvgPool3d_stride1_pad0_gpu_input"),
            ORT_TSTR("BatchNorm1d_3d_input_eval"),
            ORT_TSTR("BatchNorm2d_eval"),
            ORT_TSTR("BatchNorm2d_momentum_eval"),
            ORT_TSTR("BatchNorm3d_eval"),
            ORT_TSTR("BatchNorm3d_momentum_eval"),
            ORT_TSTR("GLU"),
            ORT_TSTR("GLU_dim"),
            ORT_TSTR("Linear"),
            ORT_TSTR("PReLU_1d"),
            ORT_TSTR("PReLU_1d_multiparam"),
            ORT_TSTR("PReLU_2d"),
            ORT_TSTR("PReLU_2d_multiparam"),
            ORT_TSTR("PReLU_3d"),
            ORT_TSTR("PReLU_3d_multiparam"),
            ORT_TSTR("PoissonNLLLLoss_no_reduce"),
            ORT_TSTR("Softsign"),
            ORT_TSTR("operator_add_broadcast"),
            ORT_TSTR("operator_add_size1_broadcast"),
            ORT_TSTR("operator_add_size1_right_broadcast"),
            ORT_TSTR("operator_add_size1_singleton_broadcast"),
            ORT_TSTR("operator_addconstant"),
            ORT_TSTR("operator_addmm"),
            ORT_TSTR("operator_basic"),
            ORT_TSTR("operator_mm"),
            ORT_TSTR("operator_non_float_params"),
            ORT_TSTR("operator_params"),
            ORT_TSTR("operator_pow"),
            ORT_TSTR("bernoulli"),
            ORT_TSTR("bernoulli_double"),
            ORT_TSTR("bernoulli_seed")};

    // float 8 types are not supported by any language.
    static const ORTCHAR_T* float8_tests[] = {
        ORT_TSTR("cast_FLOAT16_to_FLOAT8E4M3FN"),
        ORT_TSTR("cast_FLOAT16_to_FLOAT8E4M3FNUZ"),
        ORT_TSTR("cast_FLOAT16_to_FLOAT8E5M2"),
        ORT_TSTR("cast_FLOAT16_to_FLOAT8E5M2FNUZ"),
        ORT_TSTR("cast_FLOAT8E4M3FNUZ_to_FLOAT"),
        ORT_TSTR("cast_FLOAT8E4M3FNUZ_to_FLOAT16"),
        ORT_TSTR("cast_FLOAT8E4M3FN_to_FLOAT"),
        ORT_TSTR("cast_FLOAT8E4M3FN_to_FLOAT16"),
        ORT_TSTR("cast_FLOAT8E5M2FNUZ_to_FLOAT"),
        ORT_TSTR("cast_FLOAT8E5M2FNUZ_to_FLOAT16"),
        ORT_TSTR("cast_FLOAT8E5M2_to_FLOAT"),
        ORT_TSTR("cast_FLOAT8E5M2_to_FLOAT16"),
        ORT_TSTR("cast_FLOAT_to_FLOAT8E4M3FN"),
        ORT_TSTR("cast_FLOAT_to_FLOAT8E4M3FNUZ"),
        ORT_TSTR("cast_FLOAT_to_FLOAT8E5M2"),
        ORT_TSTR("cast_FLOAT_to_FLOAT8E5M2FNUZ"),
        ORT_TSTR("cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN"),
        ORT_TSTR("cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ"),
        ORT_TSTR("cast_no_saturate_FLOAT16_to_FLOAT8E5M2"),
        ORT_TSTR("cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ"),
        ORT_TSTR("cast_no_saturate_FLOAT_to_FLOAT8E4M3FN"),
        ORT_TSTR("cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ"),
        ORT_TSTR("cast_no_saturate_FLOAT_to_FLOAT8E5M2"),
        ORT_TSTR("cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ"),
        ORT_TSTR("castlike_FLOAT8E4M3FNUZ_to_FLOAT"),
        ORT_TSTR("castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded"),
        ORT_TSTR("castlike_FLOAT8E4M3FN_to_FLOAT"),
        ORT_TSTR("castlike_FLOAT8E4M3FN_to_FLOAT_expanded"),
        ORT_TSTR("castlike_FLOAT8E5M2FNUZ_to_FLOAT"),
        ORT_TSTR("castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded"),
        ORT_TSTR("castlike_FLOAT8E5M2_to_FLOAT"),
        ORT_TSTR("castlike_FLOAT8E5M2_to_FLOAT_expanded"),
        ORT_TSTR("castlike_FLOAT_to_BFLOAT16"),
        ORT_TSTR("castlike_FLOAT_to_BFLOAT16_expanded"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E4M3FN"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E4M3FNUZ"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E4M3FN_expanded"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E5M2"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E5M2FNUZ"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded"),
        ORT_TSTR("castlike_FLOAT_to_FLOAT8E5M2_expanded"),
        ORT_TSTR("dequantizelinear_e4m3fn"),
        ORT_TSTR("dequantizelinear_e5m2"),
        ORT_TSTR("quantizelinear_e4m3fn"),
        ORT_TSTR("quantizelinear_e5m2")};

    static const ORTCHAR_T* cuda_flaky_tests[] = {
        ORT_TSTR("fp16_inception_v1"),
        ORT_TSTR("fp16_shufflenet"), ORT_TSTR("fp16_tiny_yolov2")};
    static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mlperf_ssd_mobilenet_300"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("zfnet512"), ORT_TSTR("keras2coreml_Dense_ImageNet")};
    static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("test_densenet121"), ORT_TSTR("test_resnet18v2"), ORT_TSTR("test_resnet34v2"), ORT_TSTR("test_resnet50v2"), ORT_TSTR("test_resnet101v2"),
                                                     ORT_TSTR("test_resnet101v2"), ORT_TSTR("test_vgg19"), ORT_TSTR("tf_inception_resnet_v2"), ORT_TSTR("tf_inception_v1"), ORT_TSTR("tf_inception_v3"), ORT_TSTR("tf_inception_v4"), ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                     ORT_TSTR("tf_mobilenet_v2_1.0_224"), ORT_TSTR("tf_mobilenet_v2_1.4_224"), ORT_TSTR("tf_nasnet_large"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("tf_resnet_v1_50"), ORT_TSTR("tf_resnet_v1_101"), ORT_TSTR("tf_resnet_v1_101"),
                                                     ORT_TSTR("tf_resnet_v2_101"), ORT_TSTR("tf_resnet_v2_152"), ORT_TSTR("batchnorm_example_training_mode"), ORT_TSTR("batchnorm_epsilon_training_mode")};
    static const ORTCHAR_T* qnn_disabled_tests[] = {
        ORT_TSTR("nllloss_NCd1d2d3_none_no_weight_negative_ii"),
        ORT_TSTR("nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii_expanded"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii_log_prob"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded"),
        ORT_TSTR("gather_negative_indices"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded"),
        ORT_TSTR("nllloss_NCd1d2_with_weight"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_expanded"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum_expanded"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum_ii"),
        ORT_TSTR("nllloss_NCd1_weight_ii_expanded"),
        ORT_TSTR("nllloss_NCd1_ii_expanded"),
        ORT_TSTR("nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded"),
        ORT_TSTR("sce_none_weights"),
        ORT_TSTR("sce_none_weights_log_prob"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii_log_prob"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii_expanded"),
        ORT_TSTR("sce_none_weights_log_prob_expanded"),
        ORT_TSTR("sce_none_weights_expanded"),
        ORT_TSTR("convtranspose_3d"),
        ORT_TSTR("gather_elements_negative_indices"),
        ORT_TSTR("rotary_embedding_3d_input_expanded"),
        ORT_TSTR("rotary_embedding_expanded"),
        ORT_TSTR("rotary_embedding_interleaved_expanded"),
        // QNN don't support fmod = 1
        ORT_TSTR("mod_mixed_sign_float64"),
        ORT_TSTR("mod_mixed_sign_float32"),
        ORT_TSTR("mod_mixed_sign_float16"),
        ORT_TSTR("mod_int64_fmod"),
        // QNN lowers mod to cast -> a - b * floor(a/b) -> cast. Cast doesnt support these types
        ORT_TSTR("mod_mixed_sign_int16"),
        ORT_TSTR("mod_mixed_sign_int8"),
        ORT_TSTR("mod_uint16"),
        ORT_TSTR("mod_uint64")};

    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests), std::end(immutable_broken_tests));

    if (test_config.enable_cuda) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    }
    if (test_config.enable_dml) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    }
    if (test_config.enable_dnnl) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
      all_disabled_tests.insert(std::begin(float8_tests), std::end(float8_tests));
    }
    if (test_config.enable_qnn) {
      all_disabled_tests.insert(std::begin(qnn_disabled_tests), std::end(qnn_disabled_tests));
      all_disabled_tests.insert(std::begin(float8_tests), std::end(float8_tests));
    }
#if !defined(__amd64__) && !defined(_M_AMD64)
    // out of memory
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mask_rcnn_keras"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("vgg19"), ORT_TSTR("coreml_VGG16_ImageNet")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

    auto broken_tests = GetBrokenTests(test_config.provider_name);
    auto broken_tests_keyword_set = GetBrokenTestsKeyWordSet(test_config.provider_name);
    std::vector<ITestCase*> tests;
    LoadTests(test_config.data_dirs, test_config.whitelisted_test_cases,
              LoadTestTolerances(test_config.enable_cuda, test_config.enable_openvino, test_config.override_tolerance, test_config.atol, test_config.rtol),
              all_disabled_tests,
              std::move(broken_tests),
              std::move(broken_tests_keyword_set),
              [&owned_tests, &tests](std::unique_ptr<ITestCase> l) {
                tests.push_back(l.get());
                owned_tests.push_back(std::move(l));
              });

    auto tp = TestEnv::CreateThreadPool(Env::Default());
    TestEnv test_env(env, sf, tp.get(), std::move(tests), stat, test_config.inference_mode);
    Status st = test_env.Run(test_config.p_models, test_config.concurrent_session_runs, test_config.repeat_count);
    if (!st.IsOK()) {
      fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
      return -1;
    }
    std::string res = stat.ToString();
    fwrite(res.c_str(), 1, res.size(), stdout);
  }

  int result = 0;
  for (const auto& p : stat.GetFailedTest()) {
    fprintf(stderr, "test %s failed, please fix it\n", p.first.c_str());
    result = -1;
  }
  return result;
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
#ifdef _WIN32
#if defined(_DEBUG) && !defined(ONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  tmpFlag |= _CRTDBG_ALLOC_MEM_DF;
  _CrtSetDbgFlag(tmpFlag);
  std::cout << "CRT Debug Memory Leak Detection Enabled." << std::endl;
#endif
#endif

  Ort::Env env{nullptr};
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv, env);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();
  return retval;
}
