// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "command_args_parser.h"

#include <string.h>
#include <iostream>

// Windows Specific
#ifdef _WIN32
#include "getopt.h"
#include "windows.h"
#else
#include <unistd.h>
#endif

#include <core/graph/constants.h>
#include <core/framework/path_lib.h>
#include <core/optimizer/graph_transformer_level.h>

#include "test_configuration.h"

namespace onnxruntime {
namespace perftest {

/*static*/ void CommandLineParser::ShowUsage() {
  printf(
      "perf_test [options...] model_path result_file\n"
      "Options:\n"
      "\t-m [test_mode]: Specifies the test mode. Value could be 'duration' or 'times'.\n"
      "\t\tProvide 'duration' to run the test for a fix duration, and 'times' to repeated for a certain times. "
      "\t-M: Disable memory pattern.\n"
      "\t-A: Disable memory arena\n"
      "\t-c [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.\n"
      "\t-e [cpu|cuda|mkldnn|tensorrt|ngraph|openvino]: Specifies the provider 'cpu','cuda','mkldnn','tensorrt', "
      "'ngraph' or 'openvino'. "
      "Default:'cpu'.\n"
      "\t-b [tf|ort]: backend to use. Default:ort\n"
      "\t-r [repeated_times]: Specifies the repeated times if running in 'times' test mode.Default:1000.\n"
      "\t-t [seconds_to_run]: Specifies the seconds to run for 'duration' mode. Default:600.\n"
      "\t-p [profile_file]: Specifies the profile name to enable profiling and dump the profile data to the file.\n"
      "\t-s: Show statistics result, like P75, P90.\n"
      "\t-v: Show verbose information.\n"
      "\t-x [thread_size]: Session thread pool size, must >=0.\n"
      "\t-P: Use parallel executor instead of sequential executor.\n"
      "\t-o [optimization level]: Default is 1. Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).\n"
      "\t\tPlease see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels. \n"
      "\t-h: help\n");
}

/*static*/ bool CommandLineParser::ParseArguments(PerformanceTestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  int ch;
  while ((ch = getopt(argc, argv, ORT_TSTR("b:m:e:r:t:p:x:c:o:AMPvhs"))) != -1) {
    switch (ch) {
      case 'm':
        if (!CompareCString(optarg, ORT_TSTR("duration"))) {
          test_config.run_config.test_mode = TestMode::kFixDurationMode;
        } else if (!CompareCString(optarg, ORT_TSTR("times"))) {
          test_config.run_config.test_mode = TestMode::KFixRepeatedTimesMode;
        } else {
          return false;
        }
        break;
      case 'b':
        test_config.backend = optarg;
        break;
      case 'p':
        test_config.run_config.profile_file = optarg;
        break;
      case 'M':
        test_config.run_config.enable_memory_pattern = false;
        break;
      case 'A':
        test_config.run_config.enable_cpu_mem_arena = false;
        break;
      case 'e':
        if (!CompareCString(optarg, ORT_TSTR("cpu"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kCpuExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("cuda"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kCudaExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("mkldnn"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kMklDnnExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("ngraph"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kNGraphExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("brainslice"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kBrainSliceExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kTensorrtExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kOpenVINOExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("nnapi"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kNnapiExecutionProvider;
        } else {
          return false;
        }
        break;
      case 'r':
        test_config.run_config.repeated_times = static_cast<size_t>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        if (test_config.run_config.repeated_times <= 0) {
          return false;
        }
        test_config.run_config.test_mode = TestMode::KFixRepeatedTimesMode;
        break;
      case 't':
        test_config.run_config.duration_in_seconds = static_cast<size_t>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        if (test_config.run_config.repeated_times <= 0) {
          return false;
        }
        test_config.run_config.test_mode = TestMode::kFixDurationMode;
        break;
      case 's':
        test_config.run_config.f_dump_statistics = true;
        break;
      case 'v':
        test_config.run_config.f_verbose = true;
        break;
      case 'x':
        test_config.run_config.session_thread_pool_size = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        if (test_config.run_config.session_thread_pool_size < 0) {
          return false;
        }
        break;
      case 'P':
        test_config.run_config.enable_sequential_execution = false;
        break;
      case 'c':
        test_config.run_config.concurrent_session_runs =
            static_cast<size_t>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        if (test_config.run_config.concurrent_session_runs <= 0) {
          return false;
        }
        break;
      case 'o': {
        int tmp = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        switch (tmp) {
          case ORT_DISABLE_ALL:
            test_config.run_config.optimization_level = ORT_DISABLE_ALL;
            break;
          case ORT_ENABLE_BASIC:
            test_config.run_config.optimization_level = ORT_ENABLE_BASIC;
            break;
          case ORT_ENABLE_EXTENDED:
            test_config.run_config.optimization_level = ORT_ENABLE_EXTENDED;
            break;
          case ORT_ENABLE_ALL:
            test_config.run_config.optimization_level = ORT_ENABLE_ALL;
            break;
          default: {
            if (tmp > ORT_ENABLE_ALL) {  // relax constraint
              test_config.run_config.optimization_level = ORT_ENABLE_ALL;
            } else {
              return false;
            }
          }
        }
        break;
      }
      case '?':
      case 'h':
      default:
        return false;
    }
  }

  // parse model_path and result_file_path
  argc -= optind;
  argv += optind;
  if (argc != 2) return false;

  test_config.model_info.model_file_path = argv[0];
  test_config.model_info.result_file_path = argv[1];

  return true;
}

}  // namespace perftest
}  // namespace onnxruntime
