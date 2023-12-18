// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#include <thread>
#endif
#include "TestResultStat.h"
#include "TestCase.h"
#include "testenv.h"
#include "providers.h"

#include <google/protobuf/stubs/common.h>
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/framework/session_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "nlohmann/json.hpp"

using namespace onnxruntime;

namespace {
void usage() {
  auto version_string = Ort::GetVersionString();
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-A : Disable memory arena\n"
      "\t-M : Disable memory pattern\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-r [repeat]: Specifies the number of times to repeat\n"
      "\t-v: verbose\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'dnnl', 'tensorrt', "
      "'openvino', 'rocm', 'migraphx', 'acl', 'armnn', 'xnnpack', 'nnapi', 'qnn', 'snpe' or 'coreml'. "
      "Default: 'cpu'.\n"
      "\t-p: Pause after launch, can attach debugger and continue\n"
      "\t-x: Use parallel executor, default (without -x): sequential executor.\n"
      "\t-d [device_id]: Specifies the device id for multi-device (e.g. GPU). The value should > 0\n"
      "\t-t: Specify custom relative tolerance values for output value comparison. default: 1e-5\n"
      "\t-a: Specify custom absolute tolerance values for output value comparison. default: 1e-5\n"
      "\t-i: Specify EP specific runtime options as key value pairs. Different runtime options available are: \n"
      "\t    [QNN only] [backend_path]: QNN backend path. e.g '/folderpath/libQnnHtp.so', '/folderpath/libQnnCpu.so'.\n"
      "\t    [QNN only] [profiling_level]: QNN profiling level, options:  'basic', 'detailed', default 'off'.\n"
      "\t    [QNN only] [rpc_control_latency]: QNN rpc control latency. default to 10.\n"
      "\t    [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).\n"
      "\t    [QNN only] [htp_performance_mode]: QNN performance mode, options: 'burst', 'balanced', 'default', 'high_performance', \n"
      "\t    'high_power_saver', 'low_balanced', 'low_power_saver', 'power_saver', 'sustained_high_performance'. Default to 'default'. \n"
      "\t    [QNN only] [qnn_context_priority]: QNN context priority, options: 'low', 'normal', 'normal_high', 'high'. Default to 'normal'. \n"
      "\t    0 means dump the QNN context binary into separate bin file and set the path in the Onnx skeleton model.\n"
      "\t    [QNN only] [qnn_saver_path]: QNN Saver backend path. e.g '/folderpath/libQnnSaver.so'.\n"
      "\t    [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: \n"
      "\t    '0', '1', '2', '3', default is '0'.\n"
      "\t [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>' \n\n"
      "\t [Example] [For QNN EP] -e qnn -i \"profiling_level|detailed backend_path|/folderpath/libQnnCpu.so\" \n\n"
      "\t    [SNPE only] [runtime]: SNPE runtime, options: 'CPU', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n"
      "\t    [SNPE only] [priority]: execution priority, options: 'low', 'normal'. \n"
      "\t    [SNPE only] [buffer_type]: options: 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. default: ITENSOR'. \n"
      "\t    [SNPE only] [enable_init_cache]: enable SNPE init caching feature, set to 1 to enabled it. Disabled by default. \n"
      "\t [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>' \n\n"
      "\t [Example] [For SNPE EP] -e snpe -i \"runtime|CPU priority|low\" \n\n"
      "\t-o [optimization level]: Default is 99. Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).\n"
      "\t\tPlease see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels. "
      "\t-f: Enable EP context cache generation.\n"
      "\t-b: Disable EP context embed mode.\n"
      "\n"
      "\t-h: help\n"
      "\n"
      "onnxruntime version: %s\n",
      version_string.c_str());
}

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

#ifdef _WIN32
int GetNumCpuCores() {
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    // try GetSystemInfo
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (sysInfo.dwNumberOfProcessors <= 0) {
      ORT_THROW("Fatal error: 0 count processors from GetSystemInfo");
    }
    // This is the number of logical processors in the current group
    return sysInfo.dwNumberOfProcessors;
  }
  int processorCoreCount = 0;
  int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  for (int i = 0; i != count; ++i) {
    if (buffer[i].Relationship == RelationProcessorCore) {
      ++processorCoreCount;
    }
  }
  if (!processorCoreCount) ORT_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
  return processorCoreCount;
}
#else
int GetNumCpuCores() { return static_cast<int>(std::thread::hardware_concurrency()); }
#endif
}  // namespace

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[], Ort::Env& env) {
#else
int real_main(int argc, char* argv[], Ort::Env& env) {
#endif
  // if this var is not empty, only run the tests with name in this list
  std::vector<std::basic_string<PATH_CHAR_TYPE>> whitelisted_test_cases;
  int concurrent_session_runs = GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  int repeat_count = 1;
  int p_models = GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_dnnl = false;
  bool enable_openvino = false;
  bool enable_tensorrt = false;
  bool enable_mem_pattern = true;
  bool enable_qnn = false;
  bool enable_nnapi = false;
  bool enable_coreml = false;
  bool enable_snpe = false;
  bool enable_dml = false;
  bool enable_acl = false;
  bool enable_armnn = false;
  bool enable_rocm = false;
  bool enable_migraphx = false;
  bool enable_xnnpack = false;
  bool override_tolerance = false;
  double atol = 1e-5;
  double rtol = 1e-5;
  int device_id = 0;
  GraphOptimizationLevel graph_optimization_level = ORT_ENABLE_ALL;
  bool user_graph_optimization_level_set = false;
  bool set_denormal_as_zero = false;
  std::basic_string<ORTCHAR_T> ep_runtime_config_string;
  std::string provider_name = "cpu";

  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_ERROR;
  bool verbose_logging_required = false;
  bool ep_context_enable = false;
  bool disable_ep_context_embed_mode = false;

  bool pause = false;
  {
    int ch;
    while ((ch = getopt(argc, argv, ORT_TSTR("Ac:hj:Mn:r:e:t:a:xvo:d:i:pzfb"))) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'v':
          verbose_logging_required = true;
          break;
        case 'c':
          concurrent_session_runs = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (concurrent_session_runs <= 0) {
            usage();
            return -1;
          }
          break;
        case 'j':
          p_models = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (p_models <= 0) {
            usage();
            return -1;
          }
          break;
        case 'r':
          repeat_count = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (repeat_count <= 0) {
            usage();
            return -1;
          }
          break;
        case 'M':
          enable_mem_pattern = false;
          break;
        case 'n':
          // run only some whitelisted tests
          // TODO: parse name str to an array
          whitelisted_test_cases.emplace_back(optarg);
          break;
        case 'e':
          provider_name = ToUTF8String(optarg);
          if (!CompareCString(optarg, ORT_TSTR("cpu"))) {
            // do nothing
          } else if (!CompareCString(optarg, ORT_TSTR("cuda"))) {
            enable_cuda = true;
          } else if (!CompareCString(optarg, ORT_TSTR("dnnl"))) {
            enable_dnnl = true;
          } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
            enable_openvino = true;
          } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
            enable_tensorrt = true;
          } else if (!CompareCString(optarg, ORT_TSTR("qnn"))) {
            enable_qnn = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nnapi"))) {
            enable_nnapi = true;
          } else if (!CompareCString(optarg, ORT_TSTR("coreml"))) {
            enable_coreml = true;
          } else if (!CompareCString(optarg, ORT_TSTR("snpe"))) {
            enable_snpe = true;
          } else if (!CompareCString(optarg, ORT_TSTR("dml"))) {
            enable_dml = true;
          } else if (!CompareCString(optarg, ORT_TSTR("acl"))) {
            enable_acl = true;
          } else if (!CompareCString(optarg, ORT_TSTR("armnn"))) {
            enable_armnn = true;
          } else if (!CompareCString(optarg, ORT_TSTR("rocm"))) {
            enable_rocm = true;
          } else if (!CompareCString(optarg, ORT_TSTR("migraphx"))) {
            enable_migraphx = true;
          } else if (!CompareCString(optarg, ORT_TSTR("xnnpack"))) {
            enable_xnnpack = true;
          } else {
            usage();
            return -1;
          }
          break;
        case 't':
          override_tolerance = true;
          rtol = OrtStrtod<PATH_CHAR_TYPE>(optarg, nullptr);
          break;
        case 'a':
          override_tolerance = true;
          atol = OrtStrtod<PATH_CHAR_TYPE>(optarg, nullptr);
          break;
        case 'x':
          execution_mode = ExecutionMode::ORT_PARALLEL;
          break;
        case 'p':
          pause = true;
          break;
        case 'o': {
          int tmp = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          switch (tmp) {
            case ORT_DISABLE_ALL:
              graph_optimization_level = ORT_DISABLE_ALL;
              break;
            case ORT_ENABLE_BASIC:
              graph_optimization_level = ORT_ENABLE_BASIC;
              break;
            case ORT_ENABLE_EXTENDED:
              graph_optimization_level = ORT_ENABLE_EXTENDED;
              break;
            case ORT_ENABLE_ALL:
              graph_optimization_level = ORT_ENABLE_ALL;
              break;
            default: {
              if (tmp > ORT_ENABLE_ALL) {  // relax constraint
                graph_optimization_level = ORT_ENABLE_ALL;
              } else {
                fprintf(stderr, "See usage for valid values of graph optimization level\n");
                usage();
                return -1;
              }
            }
          }
          user_graph_optimization_level_set = true;
          break;
        }
        case 'd':
          device_id = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (device_id < 0) {
            usage();
            return -1;
          }
          break;
        case 'i':
          ep_runtime_config_string = optarg;
          break;
        case 'z':
          set_denormal_as_zero = true;
          break;
        case 'b':
          disable_ep_context_embed_mode = true;
          break;
        case 'f':
          ep_context_enable = true;
          break;
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }

  // TODO: Support specifying all valid levels of logging
  // Currently the logging level is ORT_LOGGING_LEVEL_ERROR by default and
  // if the user adds -v, the logging level is ORT_LOGGING_LEVEL_VERBOSE
  if (verbose_logging_required) {
    logging_level = ORT_LOGGING_LEVEL_VERBOSE;
  }

  if (concurrent_session_runs > 1 && repeat_count > 1) {
    fprintf(stderr, "when you use '-r [repeat]', please set '-c' to 1\n");
    usage();
    return -1;
  }
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify a test data dir\n");
    usage();
    return -1;
  }

  if (pause) {
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

  std::vector<std::basic_string<PATH_CHAR_TYPE>> data_dirs;
  TestResultStat stat;

  for (int i = 0; i != argc; ++i) {
    data_dirs.emplace_back(argv[i]);
  }

  std::vector<std::unique_ptr<ITestCase>> owned_tests;
  {
    Ort::SessionOptions sf;

    if (enable_cpu_mem_arena)
      sf.EnableCpuMemArena();
    else
      sf.DisableCpuMemArena();
    if (enable_mem_pattern)
      sf.EnableMemPattern();
    else
      sf.DisableMemPattern();
    sf.SetExecutionMode(execution_mode);
    if (set_denormal_as_zero)
      sf.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

    if (ep_context_enable)
      sf.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    if (disable_ep_context_embed_mode)
      sf.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");

    if (enable_tensorrt) {
#ifdef USE_TENSORRT
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = device_id;
      cuda_options.do_copy_in_default_stream = true;
      // TODO: Support arena configuration for users of test runner
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
      sf.AppendExecutionProvider_CUDA(cuda_options);
#else
      fprintf(stderr, "TensorRT is not supported in this build");
      return -1;
#endif
    }
    if (enable_openvino) {
#ifdef USE_OPENVINO
      // Setting default optimization level for OpenVINO can be overridden with -o option
      sf.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      sf.AppendExecutionProvider_OpenVINO(OrtOpenVINOProviderOptions{});
#else
      fprintf(stderr, "OpenVINO is not supported in this build");
      return -1;
#endif
    }
    if (enable_cuda) {
#ifdef USE_CUDA
      OrtCUDAProviderOptions cuda_options;
      cuda_options.do_copy_in_default_stream = true;
      // TODO: Support arena configuration for users of test runner
      sf.AppendExecutionProvider_CUDA(cuda_options);
#else
      fprintf(stderr, "CUDA is not supported in this build");
      return -1;
#endif
    }
    if (enable_dnnl) {
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
    if (enable_qnn) {
#ifdef USE_QNN
#ifdef _MSC_VER
      std::string option_string = ToUTF8String(ep_runtime_config_string);
#else
      std::string option_string = ep_runtime_config_string;
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

        if (key == "backend_path") {
          if (value.empty()) {
            ORT_THROW("Please provide the QNN backend path.");
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
        } else if (key == "rpc_control_latency" || key == "vtcm_mb") {
          // no validation
        } else if (key == "htp_performance_mode") {
          std::set<std::string> supported_htp_perf_mode = {"burst", "balanced", "default", "high_performance",
                                                           "high_power_saver", "low_balanced", "low_power_saver",
                                                           "power_saver", "sustained_high_performance"};
          if (supported_htp_perf_mode.find(value) == supported_htp_perf_mode.end()) {
            std::ostringstream str_stream;
            std::copy(supported_htp_perf_mode.begin(), supported_htp_perf_mode.end(),
                      std::ostream_iterator<std::string>(str_stream, ","));
            std::string str = str_stream.str();
            ORT_THROW("Wrong value for htp_performance_mode. select from: " + str);
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
        } else {
          ORT_THROW(R"(Wrong key type entered. Choose from options: ['backend_path',
'profiling_level', 'rpc_control_latency', 'vtcm_mb', 'htp_performance_mode',
'qnn_saver_path', 'htp_graph_finalization_optimization_mode', 'qnn_context_priority'])");
        }

        qnn_options[key] = value;
      }
      sf.AppendExecutionProvider("QNN", qnn_options);
#else
      fprintf(stderr, "QNN is not supported in this build");
      return -1;
#endif
    }
    if (enable_nnapi) {
#ifdef USE_NNAPI
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf, 0));
#else
      fprintf(stderr, "NNAPI is not supported in this build");
      return -1;
#endif
    }
    if (enable_coreml) {
#ifdef USE_COREML
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sf, 0));
#else
      fprintf(stderr, "CoreML is not supported in this build");
      return -1;
#endif
    }
    if (enable_snpe) {
#ifdef USE_SNPE
#ifdef _MSC_VER
      std::string option_string = ToUTF8String(ep_runtime_config_string);
#else
      std::string option_string = ep_runtime_config_string;
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
    if (enable_dml) {
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
    if (enable_acl) {
#ifdef USE_ACL
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "ACL is not supported in this build");
      return -1;
#endif
    }
    if (enable_armnn) {
#ifdef USE_ARMNN
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "ArmNN is not supported in this build\n");
      return -1;
#endif
    }
    if (enable_rocm) {
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
    if (enable_migraphx) {
#ifdef USE_MIGRAPHX
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sf, device_id));
#else
      fprintf(stderr, "MIGRAPHX is not supported in this build");
      return -1;
#endif
    }

    if (enable_xnnpack) {
#ifdef USE_XNNPACK
      sf.AppendExecutionProvider("XNNPACK", {});
#else
      fprintf(stderr, "XNNPACK is not supported in this build");
      return -1;
#endif
    }

    if (user_graph_optimization_level_set) {
      sf.SetGraphOptimizationLevel(graph_optimization_level);
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
        ORT_TSTR("sce_none_weights_expanded")};

    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests), std::end(immutable_broken_tests));

    if (enable_cuda) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    }
    if (enable_dml) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    }
    if (enable_dnnl) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
      all_disabled_tests.insert(std::begin(float8_tests), std::end(float8_tests));
    }
    if (enable_qnn) {
      all_disabled_tests.insert(std::begin(qnn_disabled_tests), std::end(qnn_disabled_tests));
      all_disabled_tests.insert(std::begin(float8_tests), std::end(float8_tests));
    }
#if !defined(__amd64__) && !defined(_M_AMD64)
    // out of memory
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mask_rcnn_keras"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("vgg19"), ORT_TSTR("coreml_VGG16_ImageNet")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

    auto broken_tests = GetBrokenTests(provider_name);
    auto broken_tests_keyword_set = GetBrokenTestsKeyWordSet(provider_name);
    std::vector<ITestCase*> tests;
    LoadTests(data_dirs, whitelisted_test_cases,
              LoadTestTolerances(enable_cuda, enable_openvino, override_tolerance, atol, rtol),
              all_disabled_tests,
              std::move(broken_tests),
              std::move(broken_tests_keyword_set),
              [&owned_tests, &tests](std::unique_ptr<ITestCase> l) {
                tests.push_back(l.get());
                owned_tests.push_back(std::move(l));
              });

    auto tp = TestEnv::CreateThreadPool(Env::Default());
    TestEnv test_env(env, sf, tp.get(), std::move(tests), stat);
    Status st = test_env.Run(p_models, concurrent_session_runs, repeat_count);
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
