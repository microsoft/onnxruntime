// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "ort_test_session.h"
#include <algorithm>
#include <limits>
#include <set>
#include <list>
#include <type_traits>
#include <core/session/onnxruntime_cxx_api.h>
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/providers/dnnl/dnnl_provider_options.h"
#include <assert.h>
#include "providers.h"
#include "TestCase.h"

#ifdef _WIN32
#define strdup _strdup
#endif
extern const OrtApi* g_ort;

namespace onnxruntime {
namespace perftest {

std::chrono::duration<double> OnnxRuntimeTestSession::Run() {
  // Randomly pick one OrtValueArray from test_inputs_. (NOT ThreadSafe)
  const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(test_inputs_.size() - 1));
  const size_t id = static_cast<size_t>(dist_(rand_engine_, p));
  auto& input = test_inputs_.at(id);
  auto start = std::chrono::high_resolution_clock::now();
  auto output_values = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), input.data(), input_names_.size(),
                                    output_names_raw_ptr.data(), output_names_raw_ptr.size());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - start;
  return duration_seconds;
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd,
                                               const PerformanceTestConfig& performance_test_config,
                                               const TestModelInfo& m)
    : rand_engine_(rd()), input_names_(m.GetInputCount()), input_names_str_(m.GetInputCount()), input_length_(m.GetInputCount()) {
  Ort::SessionOptions session_options;
  const std::string& provider_name = performance_test_config.machine_config.provider_type_name;
  if (provider_name == onnxruntime::kDnnlExecutionProvider) {
#ifdef USE_DNNL
    // Generate provider options
    OrtDnnlProviderOptions dnnl_options;
    dnnl_options.use_arena = 1;
    dnnl_options.threadpool_args = nullptr;

#if !defined(DNNL_ORT_THREAD)
#if defined(_MSC_VER)
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif  // defined(_MSC_VER)
    int num_threads = 0;
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if (token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW(
            "[ERROR] [OneDNN] Use a '|' to separate the key and value for the "
            "run-time option you are trying to use.\n");
      }

      auto key = token.substr(0, pos);
      auto value = token.substr(pos + 1);

      if (key == "num_of_threads") {
        std::stringstream sstream(value);
        sstream >> num_threads;
        if (num_threads < 0) {
          ORT_THROW(
              "[ERROR] [OneDNN] Invalid entry for the key 'num_of_threads',"
              " set number of threads or use '0' for default\n");
          // If the user doesnt define num_threads, auto detect threads later
        }
      } else {
        ORT_THROW(
            "[ERROR] [OneDNN] wrong key type entered. "
            "Choose from the following runtime key options that are available for OneDNN. ['num_of_threads']\n");
      }
    }
    dnnl_options.threadpool_args = static_cast<void*>(&num_threads);
#endif  // !defined(DNNL_ORT_THREAD)
    dnnl_options.use_arena = performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0;

    session_options.AppendExecutionProvider_Dnnl(dnnl_options);
#else
    ORT_THROW("DNNL is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
    const auto& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda_options;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
    std::vector<const char*> option_keys, option_values;
    // used to keep all option keys and value strings alive
    std::list<std::string> buffer;
    buffer.emplace_back("cudnn_conv_algo_search");
    option_keys.push_back(buffer.back().c_str());
    switch (performance_test_config.run_config.cudnn_conv_algo) {
      case 0:
        buffer.emplace_back("EXHAUSTIVE");
        break;
      case 1:
        buffer.emplace_back("HEURISTIC");
        break;
      default:
        buffer.emplace_back("DEFAULT");
        break;
    }
    option_values.push_back(buffer.back().c_str());

    buffer.emplace_back("do_copy_in_default_stream");
    option_keys.push_back(buffer.back().c_str());
    buffer.emplace_back(!performance_test_config.run_config.do_cuda_copy_in_separate_stream ? "1" : "0");
    option_values.push_back(buffer.back().c_str());

#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if (token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW(
            "[ERROR] [CUDA] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      buffer.emplace_back(token.substr(0, pos));
      option_keys.push_back(buffer.back().c_str());
      buffer.emplace_back(token.substr(pos + 1));
      option_values.push_back(buffer.back().c_str());
    }

    Ort::Status status(api.UpdateCUDAProviderOptions(cuda_options,
                                                     option_keys.data(), option_values.data(), option_keys.size()));
    if (!status.IsOK()) {
      OrtAllocator* allocator;
      char* options;
      Ort::ThrowOnError(api.GetAllocatorWithDefaultOptions(&allocator));
      Ort::ThrowOnError(api.GetCUDAProviderOptionsAsString(cuda_options, allocator, &options));
      ORT_THROW("[ERROR] [CUDA] Configuring the CUDA options failed with message: ", status.GetErrorMessage(),
                "\nSupported options are:\n", options);
    }
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#else
    ORT_THROW("CUDA is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
    std::vector<const char*> option_keys, option_values;
    // used to keep all option keys and value strings alive
    std::list<std::string> buffer;

#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if (token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW(
            "[ERROR] [TensorRT] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      buffer.emplace_back(token.substr(0, pos));
      option_keys.push_back(buffer.back().c_str());
      buffer.emplace_back(token.substr(pos + 1));
      option_values.push_back(buffer.back().c_str());
    }

    Ort::Status status(api.UpdateTensorRTProviderOptions(tensorrt_options,
                                                         option_keys.data(), option_values.data(), option_keys.size()));
    if (!status.IsOK()) {
      OrtAllocator* allocator;
      char* options;
      Ort::ThrowOnError(api.GetAllocatorWithDefaultOptions(&allocator));
      Ort::ThrowOnError(api.GetTensorRTProviderOptionsAsString(tensorrt_options, allocator, &options));
      ORT_THROW("[ERROR] [TensorRT] Configuring the CUDA options failed with message: ", status.GetErrorMessage(),
                "\nSupported options are:\n", options);
    }

    session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = tensorrt_options->device_id;
    cuda_options.cudnn_conv_algo_search = static_cast<OrtCudnnConvAlgoSearch>(performance_test_config.run_config.cudnn_conv_algo);
    cuda_options.do_copy_in_default_stream = !performance_test_config.run_config.do_cuda_copy_in_separate_stream;
    // TODO: Support arena configuration for users of perf test
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::unordered_map<std::string, std::string> ov_options;
    std::istringstream ss(ov_string);
    std::string token;
    while (ss >> token) {
      if (token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW("[ERROR] [OpenVINO] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      auto key = token.substr(0, pos);
      auto value = token.substr(pos + 1);

      if (key == "device_type") {
        std::set<std::string> ov_supported_device_types = {"CPU_FP32", "CPU_FP16", "GPU_FP32",
                                                           "GPU.0_FP32", "GPU.1_FP32", "GPU_FP16",
                                                           "GPU.0_FP16", "GPU.1_FP16",
                                                           "VPUX_FP16", "VPUX_U8"};
        if (ov_supported_device_types.find(value) != ov_supported_device_types.end()) {
          ov_options[key] = value;
        } else if (value.find("HETERO:") == 0) {
          ov_options[key] = value;
        } else if (value.find("MULTI:") == 0) {
          ov_options[key] = value;
        } else if (value.find("AUTO:") == 0) {
          ov_options[key] = value;
        } else {
          ORT_THROW(
              "[ERROR] [OpenVINO] You have selcted wrong configuration value for the key 'device_type'. "
              "Select from 'CPU_FP32', 'CPU_FP16', 'GPU_FP32', 'GPU.0_FP32', 'GPU.1_FP32', 'GPU_FP16', "
              "'GPU.0_FP16', 'GPU.1_FP16', 'VPUX_FP16', 'VPUX_U8' or from"
              " HETERO/MULTI/AUTO options available. \n");
        }
      } else if (key == "device_id") {
        ov_options[key] = value;
      } else if (key == "enable_vpu_fast_compile") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'enable_vpu_fast_compile' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "enable_opencl_throttling") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'enable_opencl_throttling' should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "enable_dynamic_shapes") {
        if (value == "true" || value == "True" ||
            value == "false" || value == "False") {
          ov_options[key] = value;
        } else {
          ORT_THROW(
              "[ERROR] [OpenVINO] The value for the key 'enable_dynamic_shapes' "
              "should be a boolean i.e. true or false. Default value is false.\n");
        }
      } else if (key == "num_of_threads") {
        if (std::stoi(value) <= 0) {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'num_of_threads' should be greater than 0\n");
        } else {
          ov_options[key] = value;
        }
      } else if (key == "cache_dir") {
        ov_options[key] = value;
      } else if (key == "context") {
        ov_options[key] = value;
      } else if (key == "num_streams") {
        if (std::stoi(value) <= 0 && std::stoi(value) > 8) {
          ORT_THROW("[ERROR] [OpenVINO] The value for the key 'num_streams' should be in the range of 1-8 \n");
        } else {
          ov_options[key] = value;
        }
      } else {
        ORT_THROW("[ERROR] [OpenVINO] wrong key type entered. Choose from the following runtime key options that are available for OpenVINO. ['device_type', 'device_id', 'enable_vpu_fast_compile', 'num_of_threads', 'cache_dir', 'num_streams', 'enable_opencl_throttling|true'] \n");
      }
    }
    session_options.AppendExecutionProvider("OpenVINO", ov_options);
#else
    ORT_THROW("OpenVINO is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kQnnExecutionProvider) {
#ifdef USE_QNN
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
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
      } else if (key == "qnn_context_cache_enable") {
        if (value != "1") {
          ORT_THROW("Set to 1 to enable qnn_context_cache_enable.");
        }
      } else if (key == "qnn_context_cache_path") {
        // no validation
      } else if (key == "profiling_level") {
        std::set<std::string> supported_profiling_level = {"off", "basic", "detailed"};
        if (supported_profiling_level.find(value) == supported_profiling_level.end()) {
          ORT_THROW("Supported profiling_level: off, basic, detailed");
        }
      } else if (key == "rpc_control_latency") {
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
          ORT_THROW("Supported htp_performance_mode: " + str);
        }
      } else {
        ORT_THROW(R"(Wrong key type entered. Choose from options: ['backend_path', 'qnn_context_cache_enable',
'qnn_context_cache_path', 'profiling_level', 'rpc_control_latency', 'htp_performance_mode'])");
      }

      qnn_options[key] = value;
    }
    session_options.AppendExecutionProvider("QNN", qnn_options);
#else
    ORT_THROW("QNN is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kSnpeExecutionProvider) {
#ifdef USE_SNPE
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
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
        ORT_THROW("Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
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

    session_options.AppendExecutionProvider("SNPE", snpe_options);
#else
    ORT_THROW("SNPE is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNnapiExecutionProvider) {
#ifdef USE_NNAPI
    uint32_t nnapi_flags = 0;
#ifdef _MSC_VER
    std::string ov_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string ov_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::istringstream ss(ov_string);
    std::string key;
    while (ss >> key) {
      if (key == "NNAPI_FLAG_USE_FP16") {
        nnapi_flags |= NNAPI_FLAG_USE_FP16;
      } else if (key == "NNAPI_FLAG_USE_NCHW") {
        nnapi_flags |= NNAPI_FLAG_USE_NCHW;
      } else if (key == "NNAPI_FLAG_CPU_DISABLED") {
        nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
      } else if (key == "NNAPI_FLAG_CPU_ONLY") {
        nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
      } else if (key.empty()) {
      } else {
        ORT_THROW("[ERROR] [NNAPI] wrong key type entered. Choose from the following runtime key options that are available for NNAPI. ['NNAPI_FLAG_USE_FP16', 'NNAPI_FLAG_USE_NCHW', 'NNAPI_FLAG_CPU_DISABLED', 'NNAPI_FLAG_CPU_ONLY'] \n");
      }
    }
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags));
#else
    ORT_THROW("NNAPI is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kCoreMLExecutionProvider) {
#ifdef USE_COREML
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0));
#else
    ORT_THROW("COREML is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kDmlExecutionProvider) {
#ifdef USE_DML
    std::unordered_map<std::string, std::string> dml_options;
    dml_options["performance_preference"] = "high_performance";
    dml_options["device_filter"] = "gpu";
    session_options.AppendExecutionProvider("DML", dml_options);
#else
    ORT_THROW("DML is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kAclExecutionProvider) {
#ifdef USE_ACL
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_ACL(session_options,
                                                     performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("Acl is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kArmNNExecutionProvider) {
#ifdef USE_ARMNN
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(session_options,
                                                                     performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("ArmNN is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kRocmExecutionProvider) {
#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    rocm_options.miopen_conv_exhaustive_search = performance_test_config.run_config.cudnn_conv_algo;
    rocm_options.do_copy_in_default_stream = !performance_test_config.run_config.do_cuda_copy_in_separate_stream;
    // TODO: Support arena configuration for users of perf test
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#else
    ORT_THROW("ROCM is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kMIGraphXExecutionProvider) {
#ifdef USE_MIGRAPHX
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(session_options, 0));
    OrtROCMProviderOptions rocm_options;
    rocm_options.miopen_conv_exhaustive_search = performance_test_config.run_config.cudnn_conv_algo;
    rocm_options.do_copy_in_default_stream = !performance_test_config.run_config.do_cuda_copy_in_separate_stream;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#else
    ORT_THROW("MIGraphX is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kXnnpackExecutionProvider) {
#ifdef USE_XNNPACK
    session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    session_options.AppendExecutionProvider(
        "XNNPACK", {{"intra_op_num_threads", std::to_string(performance_test_config.run_config.intra_op_num_threads)}});
#else
    ORT_THROW("Xnnpack is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kVitisAIExecutionProvider) {
#ifdef USE_VITISAI
#ifdef _MSC_VER
    std::string option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);
#else
    std::string option_string = performance_test_config.run_config.ep_runtime_config_string;
#endif
    std::istringstream ss(option_string);
    std::string token;
    std::unordered_map<std::string, std::string> vitisai_session_options;

    while (ss >> token) {
      if (token == "") {
        continue;
      }
      auto pos = token.find("|");
      if (pos == std::string::npos || pos == 0 || pos == token.length()) {
        ORT_THROW("[ERROR] [VitisAI] Use a '|' to separate the key and value for the run-time option you are trying to use.\n");
      }

      std::string key(token.substr(0, pos));
      std::string value(token.substr(pos + 1));
      vitisai_session_options[key] = value;
    }
    session_options.AppendExecutionProvider("VitisAI", vitisai_session_options);
#else
    ORT_THROW("VitisAI is not supported in this build\n");
#endif
  } else if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
    ORT_THROW("This backend is not included in perf test runner.\n");
  }

  if (performance_test_config.run_config.enable_cpu_mem_arena)
    session_options.EnableCpuMemArena();
  else
    session_options.DisableCpuMemArena();
  if (performance_test_config.run_config.enable_memory_pattern &&
      performance_test_config.run_config.execution_mode == ExecutionMode::ORT_SEQUENTIAL)
    session_options.EnableMemPattern();
  else
    session_options.DisableMemPattern();
  session_options.SetExecutionMode(performance_test_config.run_config.execution_mode);

  if (performance_test_config.run_config.intra_op_num_threads > 0) {
    fprintf(stdout, "Setting intra_op_num_threads to %d\n", performance_test_config.run_config.intra_op_num_threads);
    session_options.SetIntraOpNumThreads(performance_test_config.run_config.intra_op_num_threads);
  }

  if (!performance_test_config.run_config.intra_op_thread_affinities.empty()) {
    fprintf(stdout, "Setting intra op thread affinity as %s\n", performance_test_config.run_config.intra_op_thread_affinities.c_str());
    session_options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, performance_test_config.run_config.intra_op_thread_affinities.c_str());
  }

  if (performance_test_config.run_config.disable_spinning) {
    fprintf(stdout, "Disabling intra-op thread spinning entirely\n");
    session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
  }

  if (performance_test_config.run_config.disable_spinning_between_run) {
    fprintf(stdout, "Disabling intra-op thread spinning between runs\n");
    session_options.AddConfigEntry(kOrtSessionOptionsConfigForceSpinningStop, "1");
  }

  if (performance_test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL && performance_test_config.run_config.inter_op_num_threads > 0) {
    fprintf(stdout, "Setting inter_op_num_threads to %d\n", performance_test_config.run_config.inter_op_num_threads);
    session_options.SetInterOpNumThreads(performance_test_config.run_config.inter_op_num_threads);
  }

  // Set optimization level.
  session_options.SetGraphOptimizationLevel(performance_test_config.run_config.optimization_level);
  if (!performance_test_config.run_config.profile_file.empty())
    session_options.EnableProfiling(performance_test_config.run_config.profile_file.c_str());
  if (!performance_test_config.run_config.optimized_model_path.empty())
    session_options.SetOptimizedModelFilePath(performance_test_config.run_config.optimized_model_path.c_str());
  if (performance_test_config.run_config.set_denormal_as_zero)
    session_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");
  if (!performance_test_config.run_config.free_dim_name_overrides.empty()) {
    for (auto const& dim_override : performance_test_config.run_config.free_dim_name_overrides) {
      if (g_ort->AddFreeDimensionOverrideByName(session_options, ToUTF8String(dim_override.first).c_str(), dim_override.second) != nullptr) {
        fprintf(stderr, "AddFreeDimensionOverrideByName failed for named dimension: %s\n", ToUTF8String(dim_override.first).c_str());
      } else {
        fprintf(stdout, "Overriding dimension with name, %s, to %d\n", ToUTF8String(dim_override.first).c_str(), (int)dim_override.second);
      }
    }
  }
  if (!performance_test_config.run_config.free_dim_denotation_overrides.empty()) {
    for (auto const& dim_override : performance_test_config.run_config.free_dim_denotation_overrides) {
      if (g_ort->AddFreeDimensionOverride(session_options, ToUTF8String(dim_override.first).c_str(), dim_override.second) != nullptr) {
        fprintf(stderr, "AddFreeDimensionOverride failed for dimension denotation: %s\n", ToUTF8String(dim_override.first).c_str());
      } else {
        fprintf(stdout, "Overriding dimension with denotation, %s, to %d\n", ToUTF8String(dim_override.first).c_str(), (int)dim_override.second);
      }
    }
  }

  session_ = Ort::Session(env, performance_test_config.model_info.model_file_path.c_str(), session_options);

  size_t output_count = session_.GetOutputCount();
  output_names_.resize(output_count);
  Ort::AllocatorWithDefaultOptions a;
  for (size_t i = 0; i != output_count; ++i) {
    auto output_name = session_.GetOutputNameAllocated(i, a);
    assert(output_name != nullptr);
    output_names_[i] = output_name.get();
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }

  const size_t input_count = static_cast<size_t>(m.GetInputCount());
  for (size_t i = 0; i != input_count; ++i) {
    input_names_str_[i] = m.GetInputName(i);
    input_names_[i] = input_names_str_[i].c_str();
  }
}

template <typename T>
static void FillTensorDataTyped(Ort::Value& tensor, size_t count, int32_t seed = -1, T value = T{}) {
  T* data = tensor.GetTensorMutableData<T>();

  bool random_init = false;

  if (seed >= 0) {
    random_init = true;

    std::default_random_engine engine;
    engine.seed(seed);
    if constexpr (std::is_same<T, float>::value) {
      T max_value = 5.0f;
      const std::uniform_real_distribution<float>::param_type p(0, static_cast<float>(max_value));
      std::uniform_real_distribution<T> dist;
      for (size_t i = 0; i < count; ++i) {
        data[i] = dist(engine, p);
      }
    } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
      T max_value = std::numeric_limits<T>::max();
      const std::uniform_int_distribution<int>::param_type p(0, static_cast<int>(max_value));
      std::uniform_int_distribution<int> dist;
      for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(dist(engine, p));
      }
    } else {
      random_init = false;
      fprintf(stdout, " this type of data won't be random initialized\n");
    }
  }
  if (!random_init) {
    std::fill_n(data, count, value);
  }
}

// seed=-1 means we keep the initialized it with a constant value "T{}"
// in some case, we want to check the results for multi-runs, with the given we can recap the input data
// another reason is that, the input would be always 255/-127 for uint8_t or int8_t types of input.
// which will produce all zero outputs.
static void InitializeTensorWithSeed(int32_t seed, Ort::Value& tensor) {
  const auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
  const auto count = type_and_shape.GetElementCount();
  const auto element_type = type_and_shape.GetElementType();

#define CASE_FOR_TYPE(T)                         \
  case Ort::TypeToTensorType<T>::type: {         \
    FillTensorDataTyped<T>(tensor, count, seed); \
  } break

  switch (element_type) {
    CASE_FOR_TYPE(Ort::Float16_t);
    CASE_FOR_TYPE(Ort::BFloat16_t);
    CASE_FOR_TYPE(float);
    CASE_FOR_TYPE(double);
    CASE_FOR_TYPE(int8_t);
    CASE_FOR_TYPE(int16_t);
    CASE_FOR_TYPE(int32_t);
    CASE_FOR_TYPE(int64_t);
    CASE_FOR_TYPE(uint8_t);
    CASE_FOR_TYPE(uint16_t);
    CASE_FOR_TYPE(uint32_t);
    CASE_FOR_TYPE(uint64_t);
    CASE_FOR_TYPE(bool);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_FOR_TYPE(Ort::Float8E4M3FN_t);
    CASE_FOR_TYPE(Ort::Float8E4M3FNUZ_t);
    CASE_FOR_TYPE(Ort::Float8E5M2_t);
    CASE_FOR_TYPE(Ort::Float8E5M2FNUZ_t);
#endif
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      // string tensors are already initialized to contain empty strings
      // see onnxruntime::Tensor::Init()
      break;
    default:
      ORT_THROW("Unsupported tensor data type: ", element_type);
  }

#undef CASE_FOR_TYPE
}

bool OnnxRuntimeTestSession::PopulateGeneratedInputTestData(int32_t seed) {
  // iterate over all input nodes
  for (size_t i = 0; i < static_cast<size_t>(input_length_); i++) {
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_node_dim = tensor_info.GetShape();

      // free dimensions are treated as 1 if not overriden
      for (int64_t& dim : input_node_dim) {
        if (dim == -1) {
          dim = 1;
        }
      }

      auto allocator = Ort::AllocatorWithDefaultOptions();
      Ort::Value input_tensor = Ort::Value::CreateTensor(allocator, (const int64_t*)input_node_dim.data(),
                                                         input_node_dim.size(), tensor_info.GetElementType());
      InitializeTensorWithSeed(seed, input_tensor);
      PreLoadTestData(0, i, std::move(input_tensor));
    }
  }
  return true;
}

}  // namespace perftest
}  // namespace onnxruntime
