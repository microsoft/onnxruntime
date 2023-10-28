// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
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
#include <core/platform/path_lib.h>
#include <core/optimizer/graph_transformer_level.h>

#include "test_configuration.h"

namespace onnxruntime {
namespace perftest {

/*static*/ void CommandLineParser::ShowUsage() {
  printf(
      "perf_test [options...] model_path [result_file]\n"
      "Options:\n"
      "\t-m [test_mode]: Specifies the test mode. Value could be 'duration' or 'times'.\n"
      "\t\tProvide 'duration' to run the test for a fix duration, and 'times' to repeated for a certain times. \n"
      "\t-M: Disable memory pattern.\n"
      "\t-A: Disable memory arena\n"
      "\t-I: Generate tensor input binding (Free dimensions are treated as 1.)\n"
      "\t-c [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.\n"
      "\t-e [cpu|cuda|dnnl|tensorrt|openvino|dml|acl|nnapi|coreml|snpe|rocm|migraphx|xnnpack|vitisai]: Specifies the provider 'cpu','cuda','dnnl','tensorrt', "
      "'openvino', 'dml', 'acl', 'nnapi', 'coreml', 'snpe', 'rocm', 'migraphx', 'xnnpack' or 'vitisai'. "
      "Default:'cpu'.\n"
      "\t-b [tf|ort]: backend to use. Default:ort\n"
      "\t-r [repeated_times]: Specifies the repeated times if running in 'times' test mode.Default:1000.\n"
      "\t-t [seconds_to_run]: Specifies the seconds to run for 'duration' mode. Default:600.\n"
      "\t-p [profile_file]: Specifies the profile name to enable profiling and dump the profile data to the file.\n"
      "\t-s: Show statistics result, like P75, P90. If no result_file provided this defaults to on.\n"
      "\t-S: Given random seed, to produce the same input data. This defaults to -1(no initialize).\n"
      "\t-v: Show verbose information.\n"
      "\t-x [intra_op_num_threads]: Sets the number of threads used to parallelize the execution within nodes, A value of 0 means ORT will pick a default. Must >=0.\n"
      "\t-y [inter_op_num_threads]: Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means ORT will pick a default. Must >=0.\n"
      "\t-f [free_dimension_override]: Specifies a free dimension by name to override to a specific value for performance optimization. "
      "Syntax is [dimension_name:override_value]. override_value must > 0\n"
      "\t-F [free_dimension_override]: Specifies a free dimension by denotation to override to a specific value for performance optimization. "
      "Syntax is [dimension_denotation:override_value]. override_value must > 0\n"
      "\t-P: Use parallel executor instead of sequential executor.\n"
      "\t-o [optimization level]: Default is 99 (all). Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).\n"
      "\t\tPlease see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels.\n"
      "\t-u [optimized_model_path]: Specify the optimized model path for saving.\n"
      "\t-d [CUDA only][cudnn_conv_algorithm]: Specify CUDNN convolution algorithms: 0(benchmark), 1(heuristic), 2(default). \n"
      "\t-q [CUDA only] use separate stream for copy. \n"
      "\t-z: Set denormal as zero. When turning on this option reduces latency dramatically, a model may have denormals.\n"
      "\t-i: Specify EP specific runtime options as key value pairs. Different runtime options available are: \n"
      "\t    [DML only] [performance_preference]: DML device performance prefernce, options: 'default', 'minimum_power', 'high_performance', \n"
      "\t    [DML only] [device_filter]: DML device filter, options: 'any', 'gpu', 'npu', \n"
      "\t    [OpenVINO only] [device_type]: Overrides the accelerator hardware type and precision with these values at runtime.\n"
      "\t    [OpenVINO only] [device_id]: Selects a particular hardware device for inference.\n"
      "\t    [OpenVINO only] [enable_vpu_fast_compile]: Optionally enabled to speeds up the model's compilation on VPU device targets.\n"
      "\t    [OpenVINO only] [num_of_threads]: Overrides the accelerator hardware type and precision with these values at runtime.\n"
      "\t    [OpenVINO only] [cache_dir]: Explicitly specify the path to dump and load the blobs(Model caching) or cl_cache (Kernel Caching) files feature. If blob files are already present, it will be directly loaded.\n"
      "\t    [OpenVINO only] [enable_opencl_throttling]: Enables OpenCL queue throttling for GPU device(Reduces the CPU Utilization while using GPU) \n"
      "\t    [QNN only] [backend_path]: QNN backend path. e.g '/folderpath/libQnnHtp.so', '/folderpath/libQnnCpu.so'.\n"
      "\t    [QNN only] [qnn_context_cache_enable]: 1 to enable cache QNN context. Default to false.\n"
      "\t    [QNN only] [qnn_context_cache_path]: File path to the qnn context cache. Default to model_file.onnx.bin if not set.\n"
      "\t    [QNN only] [profiling_level]: QNN profiling level, options: 'basic', 'detailed', default 'off'.\n"
      "\t    [QNN only] [rpc_control_latency]: QNN rpc control latency. default to 10.\n"
      "\t    [QNN only] [htp_performance_mode]: QNN performance mode, options: 'burst', 'balanced', 'default', 'high_performance', \n"
      "\t    'high_power_saver', 'low_balanced', 'low_power_saver', 'power_saver', 'sustained_high_performance'. Default to 'default'. \n"
      "\t [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>'\n\n"
      "\t [Example] [For OpenVINO EP] -e openvino -i \"device_type|CPU_FP32 enable_vpu_fast_compile|true num_of_threads|5 enable_opencl_throttling|true cache_dir|\"<path>\"\"\n"
      "\t [Example] [For QNN EP] -e qnn -i \"backend_path|/folderpath/libQnnCpu.so\" \n\n"
      "\t    [TensorRT only] [trt_max_partition_iterations]: Maximum iterations for TensorRT parser to get capability.\n"
      "\t    [TensorRT only] [trt_min_subgraph_size]: Minimum size of TensorRT subgraphs.\n"
      "\t    [TensorRT only] [trt_max_workspace_size]: Set TensorRT maximum workspace size in byte.\n"
      "\t    [TensorRT only] [trt_fp16_enable]: Enable TensorRT FP16 precision.\n"
      "\t    [TensorRT only] [trt_int8_enable]: Enable TensorRT INT8 precision.\n"
      "\t    [TensorRT only] [trt_int8_calibration_table_name]: Specify INT8 calibration table name.\n"
      "\t    [TensorRT only] [trt_int8_use_native_calibration_table]: Use Native TensorRT calibration table.\n"
      "\t    [TensorRT only] [trt_dla_enable]: Enable DLA in Jetson device.\n"
      "\t    [TensorRT only] [trt_dla_core]: DLA core number.\n"
      "\t    [TensorRT only] [trt_dump_subgraphs]: Dump TRT subgraph to onnx model.\n"
      "\t    [TensorRT only] [trt_engine_cache_enable]: Enable engine caching.\n"
      "\t    [TensorRT only] [trt_engine_cache_path]: Specify engine cache path.\n"
      "\t    [TensorRT only] [trt_force_sequential_engine_build]: Force TensorRT engines to be built sequentially.\n"
      "\t    [TensorRT only] [trt_context_memory_sharing_enable]: Enable TensorRT context memory sharing between subgraphs.\n"
      "\t    [TensorRT only] [trt_layer_norm_fp32_fallback]: Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow.\n"
      "\t [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>'\n\n"
      "\t [Example] [For TensorRT EP] -e tensorrt -i 'trt_fp16_enable|true trt_int8_enable|true trt_int8_calibration_table_name|calibration.flatbuffers trt_int8_use_native_calibration_table|false trt_force_sequential_engine_build|false'\n"
      "\t    [NNAPI only] [NNAPI_FLAG_USE_FP16]: Use fp16 relaxation in NNAPI EP..\n"
      "\t    [NNAPI only] [NNAPI_FLAG_USE_NCHW]: Use the NCHW layout in NNAPI EP.\n"
      "\t    [NNAPI only] [NNAPI_FLAG_CPU_DISABLED]: Prevent NNAPI from using CPU devices.\n"
      "\t    [NNAPI only] [NNAPI_FLAG_CPU_ONLY]: Using CPU only in NNAPI EP.\n"
      "\t [Usage]: -e <provider_name> -i '<key1> <key2>'\n\n"
      "\t [Example] [For NNAPI EP] -e nnapi -i \" NNAPI_FLAG_USE_FP16 NNAPI_FLAG_USE_NCHW NNAPI_FLAG_CPU_DISABLED \"\n"
      "\t    [SNPE only] [runtime]: SNPE runtime, options: 'CPU', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n"
      "\t    [SNPE only] [priority]: execution priority, options: 'low', 'normal'. \n"
      "\t    [SNPE only] [buffer_type]: options: 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. default: ITENSOR'. \n"
      "\t    [SNPE only] [enable_init_cache]: enable SNPE init caching feature, set to 1 to enabled it. Disabled by default. \n"
      "\t [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>' \n\n"
      "\t [Example] [For SNPE EP] -e snpe -i \"runtime|CPU priority|low\" \n\n"
      "\t-T [Set intra op thread affinities]: Specify intra op thread affinity string\n"
      "\t [Example]: -T 1,2;3,4;5,6 or -T 1-2;3-4;5-6 \n"
      "\t\t Use semicolon to separate configuration between threads.\n"
      "\t\t E.g. 1,2;3,4;5,6 specifies affinities for three threads, the first thread will be attached to the first and second logical processor.\n"
      "\t\t The number of affinities must be equal to intra_op_num_threads - 1\n\n"
      "\t-D [Disable thread spinning]: disable spinning entirely for thread owned by onnxruntime intra-op thread pool.\n"
      "\t-Z [Force thread to stop spinning between runs]: disallow thread from spinning during runs to reduce cpu usage.\n"
      "\t-h: help\n");
}
#ifdef _WIN32
static const ORTCHAR_T* overrideDelimiter = L":";
#else
static const ORTCHAR_T* overrideDelimiter = ":";
#endif
static bool ParseDimensionOverride(std::basic_string<ORTCHAR_T>& dim_identifier, int64_t& override_val) {
  std::basic_string<ORTCHAR_T> free_dim_str(optarg);
  size_t delimiter_location = free_dim_str.find(overrideDelimiter);
  if (delimiter_location >= free_dim_str.size() - 1) {
    return false;
  }
  dim_identifier = free_dim_str.substr(0, delimiter_location);
  std::basic_string<ORTCHAR_T> override_val_str = free_dim_str.substr(delimiter_location + 1, std::wstring::npos);
  ORT_TRY {
    override_val = std::stoll(override_val_str.c_str());
    if (override_val <= 0) {
      return false;
    }
  }
  ORT_CATCH(...) {
    return false;
  }
  return true;
}

/*static*/ bool CommandLineParser::ParseArguments(PerformanceTestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  int ch;
  while ((ch = getopt(argc, argv, ORT_TSTR("k:b:m:e:r:t:p:x:y:c:d:o:u:i:f:F:S:T:AMPIDZvhsqz"))) != -1) {
    switch (ch) {
      case 'f': {
        std::basic_string<ORTCHAR_T> dim_name;
        int64_t override_val;
        if (!ParseDimensionOverride(dim_name, override_val)) {
          return false;
        }
        test_config.run_config.free_dim_name_overrides[dim_name] = override_val;
        break;
      }
      case 'F': {
        std::basic_string<ORTCHAR_T> dim_denotation;
        int64_t override_val;
        if (!ParseDimensionOverride(dim_denotation, override_val)) {
          return false;
        }
        test_config.run_config.free_dim_denotation_overrides[dim_denotation] = override_val;
        break;
      }
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
        } else if (!CompareCString(optarg, ORT_TSTR("dnnl"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kDnnlExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kOpenVINOExecutionProvider;
          test_config.run_config.optimization_level = ORT_DISABLE_ALL;
        } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kTensorrtExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("qnn"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kQnnExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("snpe"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kSnpeExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("nnapi"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kNnapiExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("coreml"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kCoreMLExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("dml"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kDmlExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("acl"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kAclExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("armnn"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kArmNNExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("rocm"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kRocmExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("migraphx"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kMIGraphXExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("xnnpack"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kXnnpackExecutionProvider;
        } else if (!CompareCString(optarg, ORT_TSTR("vitisai"))) {
          test_config.machine_config.provider_type_name = onnxruntime::kVitisAIExecutionProvider;
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
      case 'S':
        test_config.run_config.random_seed_for_input_data = static_cast<int32_t>(
            OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        break;
      case 'v':
        test_config.run_config.f_verbose = true;
        break;
      case 'x':
        test_config.run_config.intra_op_num_threads = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        if (test_config.run_config.intra_op_num_threads < 0) {
          return false;
        }
        break;
      case 'y':
        test_config.run_config.inter_op_num_threads = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        if (test_config.run_config.inter_op_num_threads < 0) {
          return false;
        }
        break;
      case 'P':
        test_config.run_config.execution_mode = ExecutionMode::ORT_PARALLEL;
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
      case 'k': {
        if (!CompareCString(optarg, ORT_TSTR("cpu"))) {
          test_config.run_config.native_bindings = false;
        } else if (!CompareCString(optarg, ORT_TSTR("native"))) {
          test_config.run_config.native_bindings = true;
        }
        break;
      }
      case 'u':
        test_config.run_config.optimized_model_path = optarg;
        break;
      case 'I':
        test_config.run_config.generate_model_input_binding = true;
        break;
      case 'd':
        test_config.run_config.cudnn_conv_algo = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
        break;
      case 'q':
        test_config.run_config.do_cuda_copy_in_separate_stream = true;
        break;
      case 'z':
        test_config.run_config.set_denormal_as_zero = true;
        break;
      case 'i':
        test_config.run_config.ep_runtime_config_string = optarg;
        break;
      case 'T':
        test_config.run_config.intra_op_thread_affinities = ToUTF8String(optarg);
        break;
      case 'D':
        test_config.run_config.disable_spinning = true;
        break;
      case 'Z':
        test_config.run_config.disable_spinning_between_run = true;
        break;
      case '?':
      case 'h':
      default:
        return false;
    }
  }

  // parse model_path and result_file_path
  argc -= optind;
  argv += optind;

  switch (argc) {
    case 2:
      test_config.model_info.result_file_path = argv[1];
      break;
    case 1:
      test_config.run_config.f_dump_statistics = true;
      break;
    default:
      return false;
  }

  test_config.model_info.model_file_path = argv[0];

  return true;
}

}  // namespace perftest
}  // namespace onnxruntime
