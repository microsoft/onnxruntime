// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "ort_test_session.h"
#include <algorithm>
#include <limits>
#include <fstream>
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
#include "strings_helper.h"

#if defined(USE_CUDA) || defined(USE_TENSORRT) || defined(USE_NV)
#include <cuda_runtime.h>
#endif

#ifdef USE_OPENVINO
#include "nlohmann/json.hpp"
#endif

#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#include "core/providers/dml/dml_session_options_config_keys.h"
#endif

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

  session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), input.data(), input_names_.size(),
               output_names_raw_ptr.data(), outputs_.data(), output_names_raw_ptr.size());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - start;
  return duration_seconds;
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd,
                                               const PerformanceTestConfig& performance_test_config,
                                               const TestModelInfo& m)
    : rand_engine_(rd()), input_names_(m.GetInputCount()), input_names_str_(m.GetInputCount()), input_length_(m.GetInputCount()) {
  Ort::SessionOptions session_options;

  if (!performance_test_config.selected_ep_device_indices.empty()) {
    ORT_THROW("[ERROR] [WinAppSDK]: selected_ep_device_indices is not supported.");
  }

  // Usage: --required_device_type gpu
  // has_required_device_type = false;
  // OrtHardwareDeviceType required_device_type = OrtHardwareDeviceType::OrtHardwareDeviceType_CPU;

  // Usage: -e nvtensorrtrtx
  // test_config.machine_config.provider_type_name, -e nvtensorrtrtx
  provider_name_ = performance_test_config.machine_config.provider_type_name;

  // Add EP devices if any (created by plugin EP)
  // if (performance_test_config.has_required_device_type)

  std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();
  std::vector<std::string> ep_list = performance_test_config.machine_config.plugin_provider_type_list;

  // EP -> associated EP devices (All OrtEpDevice instances must be from the same execution provider)
  std::unordered_map<std::string, std::vector<Ort::ConstEpDevice>> added_ep_devices;

  // Select EP devices by provided device index
  // if (performance_test_config.has_required_device_type) {

  for (int index = 0; index < ep_devices.size(); ++index) {
    Ort::ConstEpDevice& device = ep_devices[index];

    fprintf(stdout, "[WinML EP] EP Device [Index: %d, Name: %s]\n", static_cast<int>(index), device.EpName());

    if (device.Device().Type() == performance_test_config.required_device_type && performance_test_config.machine_config.provider_type_name == device.EpName()) {
      added_ep_devices[device.EpName()].push_back(device);
      provider_name_.append(device.EpName());
      provider_name_.append("|");
      ep_list.push_back(device.EpName());

      fprintf(stdout, "[WinML EP] EP Device [Index: %d, Name: %s] has been added to session.\n", index, device.EpName());
    }
  }

  // for (size_t index = 0; index < ep_devices.size(); ++index) {
  //   Ort::ConstEpDevice& device = ep_devices[index];
  //   if (ep_set.find(std::string(device.EpName())) != ep_set.end()) {
  //     added_ep_devices[device.EpName()].push_back(device);
  //     fprintf(stdout, "EP Device [Index: %d, Name: %s] has been added to session.\n", static_cast<int>(index), device.EpName());
  //   }
  // }

  if (added_ep_devices.empty()) {
    ORT_THROW("[ERROR] [Plugin EP]: No matching EP devices found.");
  }

  std::string ep_option_string = ToUTF8String(performance_test_config.run_config.ep_runtime_config_string);

  // EP's associated provider option lists
  std::vector<std::unordered_map<std::string, std::string>> ep_options_list;
  ParseEpOptions(ep_option_string, ep_options_list);

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
    session_options.AppendExecutionProvider_V2(env, devices, ep_options_map[ep]);
  }

  provider_name_ = performance_test_config.machine_config.provider_type_name;
  std::unordered_map<std::string, std::string> provider_options;

  if (performance_test_config.run_config.enable_cpu_mem_arena) {
    session_options.EnableCpuMemArena();
  } else {
    session_options.DisableCpuMemArena();
  }

  if (performance_test_config.run_config.enable_memory_pattern &&
      performance_test_config.run_config.execution_mode == ExecutionMode::ORT_SEQUENTIAL){
    session_options.EnableMemPattern();
  } else {
    session_options.DisableMemPattern();
  }

  session_options.SetExecutionMode(performance_test_config.run_config.execution_mode);

  // Set any extra session configuration entries provided by the user via command-line arguments.
  //
  // Some session config entries can also be set via dedicated command-line options.
  // If the user uses multiple command-line options to set the same session config entry,
  // we'll print a warning. Note that the dedicated command-line options will take precedence.
  const auto& user_session_configs = performance_test_config.run_config.session_config_entries;
  for (auto& it : user_session_configs) {
    session_options.AddConfigEntry(it.first.c_str(), it.second.c_str());
  }

  auto warn_dup_config_entry = [&user_session_configs](const char* key) -> void {
    if (user_session_configs.find(key) != user_session_configs.end()) {
      fprintf(stderr, "[WARNING]: Trying to set session config entry '%s' via multiple command-line options\n", key);
    }
  };

  if (performance_test_config.run_config.intra_op_num_threads > 0) {
    fprintf(stdout, "Setting intra_op_num_threads to %d\n", performance_test_config.run_config.intra_op_num_threads);
    session_options.SetIntraOpNumThreads(performance_test_config.run_config.intra_op_num_threads);
  }

  if (!performance_test_config.run_config.intra_op_thread_affinities.empty()) {
    warn_dup_config_entry(kOrtSessionOptionsConfigIntraOpThreadAffinities);
    fprintf(stdout, "Setting intra op thread affinity as %s\n", performance_test_config.run_config.intra_op_thread_affinities.c_str());
    session_options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, performance_test_config.run_config.intra_op_thread_affinities.c_str());
  }

  if (performance_test_config.run_config.disable_spinning) {
    warn_dup_config_entry(kOrtSessionOptionsConfigAllowIntraOpSpinning);
    fprintf(stdout, "Disabling intra-op thread spinning entirely\n");
    session_options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
  }

  if (performance_test_config.run_config.disable_spinning_between_run) {
    warn_dup_config_entry(kOrtSessionOptionsConfigForceSpinningStop);
    fprintf(stdout, "Disabling intra-op thread spinning between runs\n");
    session_options.AddConfigEntry(kOrtSessionOptionsConfigForceSpinningStop, "1");
  }

  if (!performance_test_config.run_config.register_custom_op_path.empty()) {
    session_options.RegisterCustomOpsLibrary(performance_test_config.run_config.register_custom_op_path.c_str());
  }

  if (performance_test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL && performance_test_config.run_config.inter_op_num_threads > 0) {
    fprintf(stdout, "Setting inter_op_num_threads to %d\n", performance_test_config.run_config.inter_op_num_threads);
    session_options.SetInterOpNumThreads(performance_test_config.run_config.inter_op_num_threads);
  }

  // Set optimization level.
  session_options.SetGraphOptimizationLevel(performance_test_config.run_config.optimization_level);
  if (!performance_test_config.run_config.profile_file.empty()) {
    session_options.EnableProfiling(performance_test_config.run_config.profile_file.c_str());
  }
  if (!performance_test_config.run_config.optimized_model_path.empty()) {
    session_options.SetOptimizedModelFilePath(performance_test_config.run_config.optimized_model_path.c_str());
  }
  if (performance_test_config.run_config.set_denormal_as_zero) {
    warn_dup_config_entry(kOrtSessionOptionsConfigSetDenormalAsZero);
    session_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");
  }
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

  if (performance_test_config.run_config.use_extensions) {
    session_options.EnableOrtCustomOps();
  }

  if (!performance_test_config.model_info.load_via_path) {
    session_ = Ort::Session(env, performance_test_config.model_info.model_file_path.c_str(), session_options);
  } else {
    std::ifstream file(performance_test_config.model_info.model_file_path.c_str(),
                       std::ios::binary | std::ios::in | std::ios::ate);
    if (file.is_open()) {
      const std::streampos fsize = file.tellg();
      file.seekg(0, std::ios_base::beg);
      std::vector<char> model_bytes(narrow<size_t>(fsize));
      file.read(model_bytes.data(), narrow<std::streamsize>(fsize));
      session_ = Ort::Session(env, model_bytes.data(), model_bytes.size(), session_options);
    } else {
      ORT_THROW("Model file could not be opened.\n");
    }
  }
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

  auto transform_fcn = std::function<int64_t(int64_t)>();
  auto new_value = std::function<Ort::Value(OrtAllocator*, const std::vector<int64_t>&, Ort::ConstTensorTypeAndShapeInfo&)>();
  if (device_memory_name_.empty()) {
    transform_fcn = [](int64_t input) { return input; };
    new_value = [](OrtAllocator*, const std::vector<int64_t>&, Ort::ConstTensorTypeAndShapeInfo&) {
      return Ort::Value(nullptr);
    };
  } else {
    Ort::MemoryInfo memory_info(nullptr);  // Default initialize, will be overwritten
    if (device_memory_name_ == CUDA) {
      memory_info = Ort::MemoryInfo(device_memory_name_.data(), OrtArenaAllocator, 0, OrtMemTypeDefault);
    } else {
      memory_info = Ort::MemoryInfo(device_memory_name_.data(), OrtArenaAllocator, 0, OrtMemTypeCPUOutput);
    }
    custom_allocator_ = Ort::Allocator(session_, memory_info);
    allocator_ = custom_allocator_;

    // free dimensions are treated as 1 if not overridden
    transform_fcn = [](int64_t input) { return (input == -1) ? -input : input; };
    new_value = [](OrtAllocator* allocator, const std::vector<int64_t>& output_shape, Ort::ConstTensorTypeAndShapeInfo& tensor_info) {
      return Ort::Value::CreateTensor(allocator, output_shape.data(), output_shape.size(), tensor_info.GetElementType());
    };
  }

  for (size_t i = 0; i < output_names_raw_ptr.size(); i++) {
    Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    std::transform(output_shape.begin(), output_shape.end(), output_shape.begin(), transform_fcn);
    outputs_.emplace_back(new_value(allocator_, output_shape, tensor_info));
  }

  fprintf(stdout, "[WinAppSDK] provider_names: %s\n", provider_name_.c_str());
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
  Ort::AllocatorWithDefaultOptions default_allocator;
  // iterate over all input nodes
  for (size_t i = 0; i < static_cast<size_t>(input_length_); i++) {
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_node_dim = tensor_info.GetShape();

      // free dimensions are treated as 1 if not overridden
      auto transform_fcn = [](int64_t input) { return (input == -1) ? -input : input; };
      std::transform(input_node_dim.begin(), input_node_dim.end(), input_node_dim.begin(), transform_fcn);

      if (device_memory_name_ != CUDA) {
        Ort::Value input_tensor = Ort::Value::CreateTensor(allocator_, (const int64_t*)input_node_dim.data(),
                                                           input_node_dim.size(), tensor_info.GetElementType());
        InitializeTensorWithSeed(seed, input_tensor);
        PreLoadTestData(0, i, std::move(input_tensor));
      }
// Create tensor on CPU, initialize and copy to CUDA tensor
#if defined(USE_CUDA) || defined(USE_TENSORRT) || defined(USE_NV)
      else {
        Ort::Value default_tensor = Ort::Value::CreateTensor(default_allocator, (const int64_t*)input_node_dim.data(),
                                                             input_node_dim.size(), tensor_info.GetElementType());
        InitializeTensorWithSeed(seed, default_tensor);

        // Get pointer to CPU tensor data
        const void* default_ptr = default_tensor.GetTensorRawData();

        size_t total_bytes = default_tensor.GetTensorSizeInBytes();

        Ort::Value cuda_tensor = Ort::Value::CreateTensor(allocator_, input_node_dim.data(),
                                                          input_node_dim.size(), tensor_info.GetElementType());

        void* cuda_ptr = cuda_tensor.GetTensorMutableData<void>();

        // Copy the initialized data from CPU to GPU
        cudaError_t cuda_err = cudaMemcpy(cuda_ptr, default_ptr, total_bytes, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
          ORT_THROW("Failed to copy tensor data from CPU to CUDA device. CUDA Error: ", cudaGetErrorString(cuda_err));
        }
        PreLoadTestData(0, i, std::move(cuda_tensor));
      }
#endif
    }
  }
  return true;
}

}  // namespace perftest
}  // namespace onnxruntime
