// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "command_args_parser.h"
#include "utils.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include <cxxopts.hpp>

#include <core/graph/constants.h>
#include <core/platform/path_lib.h>
#include <core/optimizer/graph_transformer_level.h>

#include "test_configuration.h"
#include "strings_helper.h"

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
      "\t-I: Generate tensor input binding. Free dimensions are treated as 1 unless overridden using -f.\n"
      "\t-c [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.\n"
      "\t-e [cpu|cuda|dnnl|tensorrt|openvino|dml|acl|nnapi|coreml|qnn|snpe|rocm|migraphx|xnnpack|vitisai|webgpu|plugin_ep]: Specifies the provider 'cpu','cuda','dnnl','tensorrt', "
      "'nvtensorrtrtx', 'openvino', 'dml', 'acl', 'nnapi', 'coreml', 'qnn', 'snpe', 'rocm', 'migraphx', 'xnnpack', 'vitisai' and 'webgpu'. "
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
      "\t-o [optimization level]: Default is 99 (all). Valid values are 0 (disable), 1 (basic), 2 (extended), 3 (layout), 99 (all).\n"
      "\t\tPlease see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels.\n"
      "\t-u [optimized_model_path]: Specify the optimized model path for saving.\n"
      "\t-d [CUDA only][cudnn_conv_algorithm]: Specify CUDNN convolution algorithms: 0(benchmark), 1(heuristic), 2(default). \n"
      "\t-q [CUDA only] use separate stream for copy. \n"
      "\t-g [TensorRT RTX | TensorRT | CUDA] Enable tensor input and output bindings on CUDA before session run \n"
      "\t-z: Set denormal as zero. When turning on this option reduces latency dramatically, a model may have denormals.\n"
      "\t-C: Specify session configuration entries as key-value pairs: -C \"<key1>|<value1> <key2>|<value2>\" \n"
      "\t    Refer to onnxruntime_session_options_config_keys.h for valid keys and values. \n"
      "\t    [Example] -C \"session.disable_cpu_ep_fallback|1 ep.context_enable|1\" \n"
      "\t-i: Specify EP specific runtime options as key value pairs. Different runtime options available are: \n"
      "\t    [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>'\n"
      "\n"
      "\t    [ACL only] [enable_fast_math]: Options: 'true', 'false', default: 'false', \n"
      "\t    [DML only] [performance_preference]: DML device performance preference, options: 'default', 'minimum_power', 'high_performance', \n"
      "\t    [DML only] [device_filter]: DML device filter, options: 'any', 'gpu', 'npu', \n"
      "\t    [DML only] [disable_metacommands]: Options: 'true', 'false', \n"
      "\t    [DML only] [enable_graph_capture]: Options: 'true', 'false', \n"
      "\t    [DML only] [enable_graph_serialization]: Options: 'true', 'false', \n"
      "\n"
      "\t    [OpenVINO only] [device_type]: Overrides the accelerator hardware type and precision with these values at runtime.\n"
      "\t    [OpenVINO only] [device_id]: Selects a particular hardware device for inference.\n"
      "\t    [OpenVINO only] [num_of_threads]: Overrides the accelerator hardware type and precision with these values at runtime.\n"
      "\t    [OpenVINO only] [cache_dir]: Explicitly specify the path to dump and load the blobs(Model caching) or cl_cache (Kernel Caching) files feature. If blob files are already present, it will be directly loaded.\n"
      "\t    [OpenVINO only] [enable_opencl_throttling]: Enables OpenCL queue throttling for GPU device(Reduces the CPU Utilization while using GPU) \n"
      "\t    [Example] [For OpenVINO EP] -e openvino -i \"device_type|CPU num_of_threads|5 enable_opencl_throttling|true cache_dir|\"<path>\"\"\n"
      "\n"
      "\t    [QNN only] [backend_type]: QNN backend type. E.g., 'cpu', 'htp'. Mutually exclusive with 'backend_path'.\n"
      "\t    [QNN only] [backend_path]: QNN backend path. E.g., '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. Mutually exclusive with 'backend_type'.\n"
      "\t    [QNN only] [profiling_level]: QNN profiling level, options: 'basic', 'detailed', default 'off'.\n"
      "\t    [QNN only] [profiling_file_path] : QNN profiling file path if ETW not enabled.\n"
      "\t    [QNN only] [rpc_control_latency]: QNN rpc control latency. default to 10.\n"
      "\t    [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).\n"
      "\t    [QNN only] [htp_performance_mode]: QNN performance mode, options: 'burst', 'balanced', 'default', 'high_performance', \n"
      "\t    'high_power_saver', 'low_balanced', 'extreme_power_saver', 'low_power_saver', 'power_saver', 'sustained_high_performance'. Default to 'default'. \n"
      "\t    [QNN only] [op_packages]: QNN UDO package, allowed format: \n"
      "\t    op_packages|<op_type>:<op_package_path>:<interface_symbol_name>[:<target>],<op_type2>:<op_package_path2>:<interface_symbol_name2>[:<target2>]. \n"
      "\t    [QNN only] [qnn_context_priority]: QNN context priority, options: 'low', 'normal', 'normal_high', 'high'. Default to 'normal'. \n"
      "\t    [QNN only] [qnn_saver_path]: QNN Saver backend path. e.g '/folderpath/libQnnSaver.so'.\n"
      "\t    [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: \n"
      "\t    '0', '1', '2', '3', default is '0'.\n"
      "\t    [QNN only] [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown). \n"
      "\t    [QNN only] [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. \n"
      "\t    Options are '0', '68', '69', '73', '75'. Defaults to '0' (none). \n"
      "\t    [QNN only] [device_id]: The ID of the device to use when setting 'htp_arch'. Defaults to '0' (for single device). \n"
      "\t    [QNN only] [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision. \n"
      "\t    Otherwise, it will be fp32 precision. Works for float32 model for HTP backend. Defaults to '1' (with FP16 precision.). \n"
      "\t    [QNN only] [offload_graph_io_quantization]: Offload graph input quantization and graph output dequantization to another EP (typically CPU EP). \n"
      "\t    Defaults to '0' (QNN EP handles the graph I/O quantization and dequantization). \n"
      "\t    [QNN only] [enable_htp_spill_fill_buffer]: Enable HTP spill fill buffer, used while generating QNN context binary.\n"
      "\t    [QNN only] [enable_htp_shared_memory_allocator]: Enable the QNN HTP shared memory allocator and use it for inputs and outputs. Requires libcdsprpc.so/dll to be available.\n"
      "\t    Defaults to '0' (disabled).\n"
      "\t    [Example] [For QNN EP] -e qnn -i \"backend_type|cpu\" \n"
      "\n"
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
      "\t    [TensorRT only] [trt_engine_cache_prefix]: Customize engine cache prefix when trt_engine_cache_enable is true.\n"
      "\t    [TensorRT only] [trt_engine_hw_compatible]: Enable hardware compatibility. Engines ending with '_sm80+' can be re-used across all Ampere+ GPU (a hardware-compatible engine may have lower throughput and/or higher latency than its non-hardware-compatible counterpart).\n"
      "\t    [TensorRT only] [trt_weight_stripped_engine_enable]: Enable weight-stripped engine build.\n"
      "\t    [TensorRT only] [trt_onnx_model_folder_path]: Folder path for the ONNX model with weights.\n"
      "\t    [TensorRT only] [trt_force_sequential_engine_build]: Force TensorRT engines to be built sequentially.\n"
      "\t    [TensorRT only] [trt_context_memory_sharing_enable]: Enable TensorRT context memory sharing between subgraphs.\n"
      "\t    [TensorRT only] [trt_layer_norm_fp32_fallback]: Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow.\n"
      "\t    [Example] [For TensorRT EP] -e tensorrt -i 'trt_fp16_enable|true trt_int8_enable|true trt_int8_calibration_table_name|calibration.flatbuffers trt_int8_use_native_calibration_table|false trt_force_sequential_engine_build|false'\n"
      "\n"
      "\t    [NNAPI only] [NNAPI_FLAG_USE_FP16]: Use fp16 relaxation in NNAPI EP..\n"
      "\t    [NNAPI only] [NNAPI_FLAG_USE_NCHW]: Use the NCHW layout in NNAPI EP.\n"
      "\t    [NNAPI only] [NNAPI_FLAG_CPU_DISABLED]: Prevent NNAPI from using CPU devices.\n"
      "\t    [NNAPI only] [NNAPI_FLAG_CPU_ONLY]: Using CPU only in NNAPI EP.\n"
      "\t    [Example] [For NNAPI EP] -e nnapi -i \"NNAPI_FLAG_USE_FP16 NNAPI_FLAG_USE_NCHW NNAPI_FLAG_CPU_DISABLED\"\n"
      "\n"
      "\t    [CoreML only] [ModelFormat]:[MLProgram, NeuralNetwork] Create an ML Program model or Neural Network. Default is NeuralNetwork.\n"
      "\t    [CoreML only] [MLComputeUnits]:[CPUAndNeuralEngine CPUAndGPU ALL CPUOnly] Specify to limit the backend device used to run the model.\n"
      "\t    [CoreML only] [AllowStaticInputShapes]:[0 1].\n"
      "\t    [CoreML only] [EnableOnSubgraphs]:[0 1].\n"
      "\t    [CoreML only] [SpecializationStrategy]:[Default FastPrediction].\n"
      "\t    [CoreML only] [ProfileComputePlan]:[0 1].\n"
      "\t    [CoreML only] [AllowLowPrecisionAccumulationOnGPU]:[0 1].\n"
      "\t    [CoreML only] [ModelCacheDirectory]:[path../a/b/c].\n"
      "\t    [Example] [For CoreML EP] -e coreml -i \"ModelFormat|MLProgram MLComputeUnits|CPUAndGPU\"\n"
      "\n"
      "\t    [SNPE only] [runtime]: SNPE runtime, options: 'CPU', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n"
      "\t    [SNPE only] [priority]: execution priority, options: 'low', 'normal'. \n"
      "\t    [SNPE only] [buffer_type]: options: 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. default: ITENSOR'. \n"
      "\t    [SNPE only] [enable_init_cache]: enable SNPE init caching feature, set to 1 to enabled it. Disabled by default. \n"
      "\t    [Example] [For SNPE EP] -e snpe -i \"runtime|CPU priority|low\" \n\n"
      "\n"
      "\t-T [Set intra op thread affinities]: Specify intra op thread affinity string\n"
      "\t [Example]: -T 1,2;3,4;5,6 or -T 1-2;3-4;5-6 \n"
      "\t\t Use semicolon to separate configuration between threads.\n"
      "\t\t E.g. 1,2;3,4;5,6 specifies affinities for three threads, the first thread will be attached to the first and second logical processor.\n"
      "\t\t The number of affinities must be equal to intra_op_num_threads - 1\n\n"
      "\t-D [Disable thread spinning]: disable spinning entirely for thread owned by onnxruntime intra-op thread pool.\n"
      "\t-Z [Force thread to stop spinning between runs]: disallow thread from spinning during runs to reduce cpu usage.\n"
      "\t-n [Exit after session creation]: allow user to measure session creation time to measure impact of enabling any initialization optimizations.\n"
      "\t-l Provide file as binary in memory by using fopen before session creation.\n"
      "\t-R [Register custom op]: allow user to register custom op by .so or .dll file.\n"
      "\t-X [Enable onnxruntime-extensions custom ops]: Registers custom ops from onnxruntime-extensions. "
      "onnxruntime-extensions must have been built in to onnxruntime. This can be done with the build.py "
      "'--use_extensions' option.\n"
      "\n"
      "\t--plugin_ep_libs      [registration names and libraries] Specifies a list of plugin execution provider (EP) registration names and their corresponding shared libraries to register.\n"
      "\t                      [Usage]: --plugin_ep_libs \"plugin_ep_name_1|plugin_ep_1.dll plugin_ep_name_2|plugin_ep_2.dll ... \"\n"
      "\n"
      "\t--plugin_eps          [Plugin EPs] Specifies a semicolon-separated list of plugin execution providers (EPs) to use.\n"
      "\t                      [Usage]: --plugin_eps \"plugin_ep_1;plugin_ep_2;... \"\n"
      "\n"
      "\t--plugin_ep_options   [EP options] Specifies provider options for each EP listed in --plugin_eps. Options (key-value pairs) for each EP are separated by space and EPs are separated by semicolons.\n"
      "\t                      [Usage]: --plugin_ep_options \"ep_1_option_1_key|ep_1_option_1_value ...;ep_2_option_1_key|ep_2_option_1_value ...;... \" or \n"
      "\t                               --plugin_ep_options \";ep_2_option_1_key|ep_2_option_1_value ...;... \" or \n"
      "\t                               --plugin_ep_options \"ep_1_option_1_key|ep_1_option_1_value ...;;ep_3_option_1_key|ep_3_option_1_value ...;... \" or \n"
      "\n"
      "\t--list_ep_devices     Prints all available device indices and their properties (including metadata). This option makes the program exit early without performing inference.\n"
      "\t--select_ep_devices   [list of device indices] A semicolon-separated list of device indices to add to the session and run with.\n"
      "\t-h: help\n");
}
#ifdef _WIN32
static const ORTCHAR_T* overrideDelimiter = L":";
#else
static const ORTCHAR_T* overrideDelimiter = ":";
#endif
static bool ParseDimensionOverride(std::basic_string<ORTCHAR_T>& dim_identifier, int64_t& override_val, const ORTCHAR_T* option) {
  std::basic_string<ORTCHAR_T> free_dim_str(option);
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

bool CommandLineParser::ParseArguments(PerformanceTestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  ORT_TRY {
    cxxopts::Options options("onnxruntime_perf_test", "perf_test [options...] model_path [result_file]");

    // See ShowUsage() for detailed option descriptions.
    options.add_options()("f", "", cxxopts::value<std::vector<std::string> >());
    options.add_options()("F", "", cxxopts::value<std::vector<std::string> >());
    options.add_options()("m", "", cxxopts::value<std::string>());
    options.add_options()("e", "", cxxopts::value<std::string>());
    options.add_options()("r", "", cxxopts::value<size_t>());
    options.add_options()("t", "", cxxopts::value<size_t>());
    options.add_options()("p", "", cxxopts::value<std::string>());
    options.add_options()("x", "", cxxopts::value<int>());
    options.add_options()("y", "", cxxopts::value<int>());
    options.add_options()("c", "", cxxopts::value<size_t>());
    options.add_options()("d", "", cxxopts::value<int>());
    options.add_options()("o", "", cxxopts::value<int>());
    options.add_options()("u", "", cxxopts::value<std::string>());
    options.add_options()("i", "", cxxopts::value<std::string>());
    options.add_options()("S", "", cxxopts::value<int>());
    options.add_options()("T", "", cxxopts::value<std::string>());
    options.add_options()("C", "", cxxopts::value<std::string>());
    options.add_options()("R", "", cxxopts::value<std::string>());
    options.add_options()("A", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("M", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("s", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("v", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("I", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("P", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("q", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("z", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("D", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("Z", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("n", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("l", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("g", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("X", "", cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
    options.add_options()("plugin_ep_libs", "", cxxopts::value<std::string>());
    options.add_options()("plugin_eps", "", cxxopts::value<std::string>());
    options.add_options()("plugin_ep_options", "", cxxopts::value<std::string>());
    options.add_options()("list_ep_devices", "");
    options.add_options()("select_ep_devices", "", cxxopts::value<std::string>());
    options.add_options()("h,help", "");

#ifdef _WIN32
    auto utf8_strings = utils::ConvertArgvToUtf8Strings(argc, argv);
    auto utf8_argv = utils::CStringsFromStrings(utf8_strings);
    auto result = options.parse(static_cast<int>(utf8_argv.size()), utf8_argv.data());
#else
    auto result = options.parse(argc, argv);
#endif

    if (result.count("f")) {
      std::basic_string<ORTCHAR_T> dim_name;
      int64_t override_val;
      std::basic_string<ORTCHAR_T> opt_str = ToPathString(result["f"].as<std::string>());
      if (!ParseDimensionOverride(dim_name, override_val, opt_str.c_str())) {
        return false;
      }
      test_config.run_config.free_dim_name_overrides[dim_name] = override_val;
    }

    if (result.count("F")) {
      std::basic_string<ORTCHAR_T> dim_denotation;
      int64_t override_val;
      std::basic_string<ORTCHAR_T> opt_str = ToPathString(result["F"].as<std::string>());
      if (!ParseDimensionOverride(dim_denotation, override_val, opt_str.c_str())) {
        return false;
      }
      test_config.run_config.free_dim_denotation_overrides[dim_denotation] = override_val;
    }

    if (result.count("m")) {
      std::basic_string<ORTCHAR_T> opt_str = ToPathString(result["m"].as<std::string>());
      if (!CompareCString(opt_str.c_str(), ORT_TSTR("duration"))) {
        test_config.run_config.test_mode = TestMode::kFixDurationMode;
      } else if (!CompareCString(opt_str.c_str(), ORT_TSTR("times"))) {
        test_config.run_config.test_mode = TestMode::KFixRepeatedTimesMode;
      } else {
        return false;
      }
    }

    if (result.count("p")) test_config.run_config.profile_file = ToPathString(result["p"].as<std::string>());
    if (result["M"].as<bool>()) test_config.run_config.enable_memory_pattern = false;
    if (result["A"].as<bool>()) test_config.run_config.enable_cpu_mem_arena = false;

    if (result.count("e")) {
      auto optarg = result["e"].as<std::string>().c_str();
      if (!CompareCString(optarg, "cpu")) {
        test_config.machine_config.provider_type_name = onnxruntime::kCpuExecutionProvider;
      } else if (!CompareCString(optarg, "cuda")) {
        test_config.machine_config.provider_type_name = onnxruntime::kCudaExecutionProvider;
      } else if (!CompareCString(optarg, "dnnl")) {
        test_config.machine_config.provider_type_name = onnxruntime::kDnnlExecutionProvider;
      } else if (!CompareCString(optarg, "openvino")) {
        test_config.machine_config.provider_type_name = onnxruntime::kOpenVINOExecutionProvider;
      } else if (!CompareCString(optarg, "tensorrt")) {
        test_config.machine_config.provider_type_name = onnxruntime::kTensorrtExecutionProvider;
      } else if (!CompareCString(optarg, "qnn")) {
        test_config.machine_config.provider_type_name = onnxruntime::kQnnExecutionProvider;
      } else if (!CompareCString(optarg, "snpe")) {
        test_config.machine_config.provider_type_name = onnxruntime::kSnpeExecutionProvider;
      } else if (!CompareCString(optarg, "nnapi")) {
        test_config.machine_config.provider_type_name = onnxruntime::kNnapiExecutionProvider;
      } else if (!CompareCString(optarg, "vsinpu")) {
        test_config.machine_config.provider_type_name = onnxruntime::kVSINPUExecutionProvider;
      } else if (!CompareCString(optarg, "coreml")) {
        test_config.machine_config.provider_type_name = onnxruntime::kCoreMLExecutionProvider;
      } else if (!CompareCString(optarg, "dml")) {
        test_config.machine_config.provider_type_name = onnxruntime::kDmlExecutionProvider;
      } else if (!CompareCString(optarg, "acl")) {
        test_config.machine_config.provider_type_name = onnxruntime::kAclExecutionProvider;
      } else if (!CompareCString(optarg, "armnn")) {
        test_config.machine_config.provider_type_name = onnxruntime::kArmNNExecutionProvider;
      } else if (!CompareCString(optarg, "rocm")) {
        test_config.machine_config.provider_type_name = onnxruntime::kRocmExecutionProvider;
      } else if (!CompareCString(optarg, "migraphx")) {
        test_config.machine_config.provider_type_name = onnxruntime::kMIGraphXExecutionProvider;
      } else if (!CompareCString(optarg, "xnnpack")) {
        test_config.machine_config.provider_type_name = onnxruntime::kXnnpackExecutionProvider;
      } else if (!CompareCString(optarg, "vitisai")) {
        test_config.machine_config.provider_type_name = onnxruntime::kVitisAIExecutionProvider;
      } else if (!CompareCString(optarg, "webgpu")) {
        test_config.machine_config.provider_type_name = onnxruntime::kWebGpuExecutionProvider;
      } else if (!CompareCString(optarg, "nvtensorrtrtx")) {
        test_config.machine_config.provider_type_name = onnxruntime::kNvTensorRTRTXExecutionProvider;
      } else {
        return false;
      }
    }

    if (result.count("r")) {
      auto val = result["r"].as<size_t>();
      if (val <= 0) return false;
      test_config.run_config.repeated_times = val;
      test_config.run_config.test_mode = TestMode::KFixRepeatedTimesMode;
    }

    if (result.count("t")) {
      auto val = result["t"].as<size_t>();
      if (val <= 0) return false;
      test_config.run_config.duration_in_seconds = val;
      test_config.run_config.test_mode = TestMode::kFixDurationMode;
    }

    if (result["s"].as<bool>()) test_config.run_config.f_dump_statistics = true;

    if (result.count("S")) {
      auto val = result["S"].as<int>();
      test_config.run_config.random_seed_for_input_data = val;
    }

    if (result["v"].as<bool>()) test_config.run_config.f_verbose = true;

    if (result.count("x")) {
      auto val = result["x"].as<int>();
      if (val < 0) return false;
      test_config.run_config.intra_op_num_threads = val;
    }

    if (result.count("y")) {
      auto val = result["y"].as<int>();
      if (val < 0) return false;
      test_config.run_config.inter_op_num_threads = val;
    }

    if (result.count("P")) {
      test_config.run_config.execution_mode = ExecutionMode::ORT_PARALLEL;
    }

    if (result.count("c")) {
      auto val = result["c"].as<size_t>();
      if (static_cast<int>(val) <= 0) return false;
      test_config.run_config.concurrent_session_runs = val;
    }

    if (result.count("o")) {
      auto val = result["o"].as<int>();
      switch (val) {
        case ORT_DISABLE_ALL:
          test_config.run_config.optimization_level = ORT_DISABLE_ALL;
          break;
        case ORT_ENABLE_BASIC:
          test_config.run_config.optimization_level = ORT_ENABLE_BASIC;
          break;
        case ORT_ENABLE_EXTENDED:
          test_config.run_config.optimization_level = ORT_ENABLE_EXTENDED;
          break;
        case ORT_ENABLE_LAYOUT:
          test_config.run_config.optimization_level = ORT_ENABLE_LAYOUT;
          break;
        case ORT_ENABLE_ALL:
          test_config.run_config.optimization_level = ORT_ENABLE_ALL;
          break;
        default: {
          if (val > ORT_ENABLE_ALL) {  // relax constraint
            test_config.run_config.optimization_level = ORT_ENABLE_ALL;
          } else {
            return false;
          }
        }
      }
    }

    if (result.count("u")) test_config.run_config.optimized_model_path = ToPathString(result["u"].as<std::string>());

    if (result.count("I")) test_config.run_config.generate_model_input_binding = true;

    if (result.count("d")) {
      auto val = result["d"].as<int>();
      if (val < 0) return false;
      test_config.run_config.cudnn_conv_algo = val;
    }

    if (result.count("q")) test_config.run_config.do_cuda_copy_in_separate_stream = true;

    if (result.count("z")) test_config.run_config.set_denormal_as_zero = true;

    if (result.count("i")) test_config.run_config.ep_runtime_config_string = ToPathString(result["i"].as<std::string>());

    if (result.count("T")) test_config.run_config.intra_op_thread_affinities = result["T"].as<std::string>();

    if (result.count("C")) {
      ORT_TRY {
        ParseSessionConfigs(result["C"].as<std::string>(), test_config.run_config.session_config_entries);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          fprintf(stderr, "Error parsing session configuration entries: %s\n", ex.what());
        });
        return false;
      }
    }

    if (result.count("D")) test_config.run_config.disable_spinning = true;

    if (result.count("Z")) test_config.run_config.disable_spinning_between_run = true;

    if (result.count("n")) test_config.run_config.exit_after_session_creation = true;

    if (result.count("l")) test_config.model_info.load_via_path = true;

    if (result.count("R")) test_config.run_config.register_custom_op_path = ToPathString(result["R"].as<std::string>());

    if (result.count("g")) test_config.run_config.enable_cuda_io_binding = true;

    if (result.count("X")) test_config.run_config.use_extensions = true;

    if (result.count("plugin_ep_libs")) test_config.plugin_ep_names_and_libs = ToPathString(result["plugin_ep_libs"].as<std::string>());

    if (result.count("plugin_eps")) ParseEpList(result["plugin_eps"].as<std::string>(), test_config.machine_config.plugin_provider_type_list);

    if (result.count("plugin_ep_options")) test_config.run_config.ep_runtime_config_string = ToPathString(result["plugin_ep_options"].as<std::string>());

    if (result.count("select_ep_devices")) test_config.selected_devices = result["select_ep_devices"].as<std::string>();

    if (result.count("h")) {
      perftest::CommandLineParser::ShowUsage();
      return false;
    }

    if (result.count("list_ep_devices")) {
      test_config.list_available_devices = true;
      return true;
    }

    // Positional arguments
    std::vector<std::string> positional = result.unmatched();
    if (positional.size() == 1) {
      test_config.model_info.model_file_path = ToPathString(positional[0]);
      test_config.run_config.f_dump_statistics = true;
    } else if (positional.size() == 2) {
      test_config.model_info.model_file_path = ToPathString(positional[0]);
      test_config.model_info.result_file_path = ToPathString(positional[1]);
    } else {
      return false;
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "Error parsing options: %s\n", ex.what());
    });
    return false;
  }

  return true;
}

}  // namespace perftest
}  // namespace onnxruntime
