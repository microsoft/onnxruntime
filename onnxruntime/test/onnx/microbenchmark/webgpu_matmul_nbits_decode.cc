// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <core/common/common.h>
#include <core/graph/onnx_protobuf.h>
#include <core/platform/env.h>
#include "core/providers/webgpu/webgpu_provider_options.h"
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include "core/session/onnxruntime_run_options_config_keys.h"

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <webgpu/webgpu.h>

extern OrtEnv* env;
extern const OrtApi* g_ort;

namespace {
constexpr const char* kDecodeBenchmarkModeEnvVar = "ORT_WEBGPU_MATMUL_NBITS_BENCHMARK_MODE";
constexpr const char* kDecodeBenchmarkGraphCaptureEnvVar = "ORT_WEBGPU_MATMUL_NBITS_ENABLE_GRAPH_CAPTURE";
constexpr const char* kDecodeBenchmarkOptimizedModelPathEnvVar = "ORT_WEBGPU_MATMUL_NBITS_OPTIMIZED_MODEL_PATH";
constexpr const char* kDecodeBenchmarkVerboseSessionLogEnvVar = "ORT_WEBGPU_MATMUL_NBITS_VERBOSE_SESSION_LOG";
constexpr float kDecodeCorrectnessAbsTolerance = 0.1f;
constexpr float kDecodeCorrectnessRelTolerance = 0.01f;
constexpr const char* kBenchmarkGraphCaptureAnnotationId = "1";

enum class DecodeBenchmarkMode {
  kPerf,
  kCorrectness,
};

bool IsGraphCaptureBenchmarkEnabled();
bool IsGraphCaptureBenchmarkEnabled();
bool IsVerboseSessionLogEnabled();
std::string GetOptimizedModelPath();

enum class MlpDecodeBenchmarkVariant {
  kUnfused,
  kFused,
};

enum class MlpNormKind {
  kSimplified,
  kSkipSimplified,
  kSkipSimplifiedPassthrough,
};

struct MlpDecodeBenchConfig {
  int64_t n;
  int64_t k;
  int64_t bits;
  int64_t block_size;
  int64_t accuracy_level;
};

struct AdapterSelectionConfig {
  // preferred_device_substring: optional case-insensitive device-name hint.
  // context_id: ORT WebGPU custom context ID used to bind the externally created instance/device.
  // backend_type: Dawn backend to enumerate adapters from, e.g. D3D12 or Vulkan.
  // print_adapter_list: whether to print all discovered adapters before selecting one.
  const char* preferred_device_substring;
  int context_id;
  WGPUBackendType backend_type;
  bool print_adapter_list;
};

struct AdapterCandidate {
  dawn::native::Adapter adapter;
  int global_index;
  WGPUAdapterType adapter_type;
  int type_index;
  uint32_t vendor_id;
  uint32_t device_id;
  std::string vendor;
  std::string architecture;
  std::string device;
  std::string description;
};

struct SelectedWebGpuContext {
  std::unique_ptr<dawn::native::Instance> dawn_instance;
  WGPUInstance instance{nullptr};
  WGPUDevice device{nullptr};
  std::unordered_map<std::string, std::string> provider_options;
  std::string selected_adapter_summary;
};

struct MlpTrafficStats {
  double input_bytes;
  double packed_weight_bytes;
  double scale_bytes;
  double intermediate_bytes;
  double output_bytes;
  double total_bytes;
};

struct QkvDecodeBenchConfig {
  int64_t q_n;
  int64_t kv_n;
  int64_t k;
  int64_t bits;
  int64_t block_size;
  int64_t accuracy_level;
};

enum class QkvDecodeBenchmarkVariant {
  kUnfused,
  kFused,
};

enum class QkvNormKind {
  kSimplified,
  kSkipSimplified,
  kSkipSimplifiedPassthrough,
};

struct QkvTrafficStats {
  double input_bytes;
  double skip_input_bytes;
  double norm_scale_bytes;
  double packed_weight_bytes;
  double scale_bytes;
  double intermediate_bytes;
  double output_bytes;
  double total_bytes;
};

constexpr double kRtxTheoreticalBandwidthBytesPerSecond = 448.0 * 1000.0 * 1000.0 * 1000.0;
constexpr int kDecodeWarmupRuns = 25;

DecodeBenchmarkMode GetDecodeBenchmarkMode() {
  std::string mode_env = onnxruntime::Env::Default().GetEnvironmentVar(kDecodeBenchmarkModeEnvVar);
  if (mode_env.empty()) {
    return DecodeBenchmarkMode::kPerf;
  }

  std::transform(mode_env.begin(), mode_env.end(), mode_env.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  if (mode_env == "0" || mode_env == "false" || mode_env == "off" ||
      mode_env == "check" || mode_env == "correctness" || mode_env == "validate") {
    return DecodeBenchmarkMode::kCorrectness;
  }

  return DecodeBenchmarkMode::kPerf;
}

bool IsDecodeBenchmarkPerfMode() {
  return GetDecodeBenchmarkMode() == DecodeBenchmarkMode::kPerf;
}

std::string GetDecodeBenchmarkLabel(const char* shape_label = nullptr) {
  const char* mode_label = IsDecodeBenchmarkPerfMode() ? "perf" : "correctness";
  const char* graph_label = IsGraphCaptureBenchmarkEnabled() ? "graph_on" : "graph_off";

  std::ostringstream stream;
  stream << "fp16_decode";
  if (shape_label != nullptr && shape_label[0] != '\0') {
    stream << '_' << shape_label;
  }
  stream << '_' << mode_label << "_auto_gpu_" << graph_label;
  return stream.str();
}

bool IsGraphCaptureBenchmarkEnabled() {
  std::string graph_capture_env = onnxruntime::Env::Default().GetEnvironmentVar(kDecodeBenchmarkGraphCaptureEnvVar);
  if (graph_capture_env.empty()) {
    return false;
  }

  std::transform(graph_capture_env.begin(), graph_capture_env.end(), graph_capture_env.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  return graph_capture_env != "0" && graph_capture_env != "false" && graph_capture_env != "off";
}

bool IsVerboseSessionLogEnabled() {
  std::string verbose_log_env = onnxruntime::Env::Default().GetEnvironmentVar(kDecodeBenchmarkVerboseSessionLogEnvVar);
  if (verbose_log_env.empty()) {
    return false;
  }

  std::transform(verbose_log_env.begin(), verbose_log_env.end(), verbose_log_env.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  return verbose_log_env != "0" && verbose_log_env != "false" && verbose_log_env != "off";
}

std::string GetOptimizedModelPath() {
  return onnxruntime::Env::Default().GetEnvironmentVar(kDecodeBenchmarkOptimizedModelPathEnvVar);
}

Ort::RunOptions CreateBenchmarkRunOptions() {
  Ort::RunOptions run_options;
  if (IsGraphCaptureBenchmarkEnabled()) {
    run_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, kBenchmarkGraphCaptureAnnotationId);
  }

  return run_options;
}

std::vector<wgpu::FeatureName> GetRequiredDeviceFeatures(const wgpu::Adapter& adapter) {
  std::vector<wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
#if !defined(__wasm__)
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
      wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix,
#endif
      wgpu::FeatureName::TimestampQuery,
      wgpu::FeatureName::ShaderF16,
      wgpu::FeatureName::Subgroups,
#if !defined(__wasm__)
      wgpu::FeatureName::BufferMapExtendedUsages,
#endif
  };
  for (auto feature : features) {
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    }
  }
  return required_features;
}

wgpu::Limits GetRequiredDeviceLimits(const wgpu::Adapter& adapter) {
  wgpu::Limits required_limits{};
  wgpu::Limits adapter_limits{};
  if (!adapter.GetLimits(&adapter_limits)) {
    throw std::runtime_error("Failed to query adapter limits for the selected WebGPU adapter.");
  }

  required_limits.maxBindGroups = adapter_limits.maxBindGroups;
  required_limits.maxComputeWorkgroupStorageSize = adapter_limits.maxComputeWorkgroupStorageSize;
  required_limits.maxComputeWorkgroupsPerDimension = adapter_limits.maxComputeWorkgroupsPerDimension;
  required_limits.maxStorageBuffersPerShaderStage = adapter_limits.maxStorageBuffersPerShaderStage;
  required_limits.maxStorageBufferBindingSize = adapter_limits.maxStorageBufferBindingSize;
  required_limits.maxBufferSize = adapter_limits.maxBufferSize;
  required_limits.maxComputeInvocationsPerWorkgroup = adapter_limits.maxComputeInvocationsPerWorkgroup;
  required_limits.maxComputeWorkgroupSizeX = adapter_limits.maxComputeWorkgroupSizeX;
  required_limits.maxComputeWorkgroupSizeY = adapter_limits.maxComputeWorkgroupSizeY;
  required_limits.maxComputeWorkgroupSizeZ = adapter_limits.maxComputeWorkgroupSizeZ;

  return required_limits;
}

std::string ToString(WGPUStringView value) {
  return value.data == nullptr ? std::string{} : std::string(value.data, value.length);
}

const char* AdapterTypeToString(WGPUAdapterType adapter_type) {
  switch (adapter_type) {
    case WGPUAdapterType_DiscreteGPU:
      return "discrete";
    case WGPUAdapterType_IntegratedGPU:
      return "integrated";
    case WGPUAdapterType_CPU:
      return "cpu";
    default:
      return "unknown";
  }
}

bool IsGpuAdapterType(WGPUAdapterType adapter_type) {
  return adapter_type == WGPUAdapterType_DiscreteGPU ||
         adapter_type == WGPUAdapterType_IntegratedGPU;
}

std::string FormatAdapterSummary(const AdapterCandidate& adapter) {
  std::ostringstream stream;
  stream << "adapter[" << adapter.global_index << "]"
         << " type=" << AdapterTypeToString(adapter.adapter_type)
         << " type_index=" << adapter.type_index
         << " vendor=" << adapter.vendor
         << " architecture=" << adapter.architecture
         << " gpu_name=" << adapter.device
         << " description=" << adapter.description
         << " vendor_id=" << adapter.vendor_id
         << " device_id=" << adapter.device_id;
  return stream.str();
}

std::string FormatFeatureSupport(const dawn::native::Adapter& adapter) {
  const wgpu::Adapter wgpu_adapter = adapter.Get();
  std::ostringstream stream;
  stream << "shader_f16=" << (wgpu_adapter.HasFeature(wgpu::FeatureName::ShaderF16) ? "yes" : "no")
         << " subgroups=" << (wgpu_adapter.HasFeature(wgpu::FeatureName::Subgroups) ? "yes" : "no")
         << " timestamp_query=" << (wgpu_adapter.HasFeature(wgpu::FeatureName::TimestampQuery) ? "yes" : "no");
#if !defined(__wasm__)
  stream << " subgroup_matrix=" << (wgpu_adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix) ? "yes" : "no")
         << " buffer_map_extended_usages=" << (wgpu_adapter.HasFeature(wgpu::FeatureName::BufferMapExtendedUsages) ? "yes" : "no")
         << " timestamp_query_inside_passes=" << (wgpu_adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses) ? "yes" : "no");
#endif
  return stream.str();
}

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
  return value;
}

MlpTrafficStats CalculateMlpTrafficStats(const MlpDecodeBenchConfig& config,
                                         MlpDecodeBenchmarkVariant variant,
                                         MlpNormKind norm_kind) {
  const int64_t k_blocks = (config.k + config.block_size - 1) / config.block_size;
  const int64_t blob_size = (config.block_size * config.bits) / 8;

  const bool is_unfused = variant == MlpDecodeBenchmarkVariant::kUnfused;
  const double input_reads = variant == MlpDecodeBenchmarkVariant::kUnfused ? 2.0 : 1.0;
  const bool has_skip = norm_kind == MlpNormKind::kSkipSimplified ||
                        norm_kind == MlpNormKind::kSkipSimplifiedPassthrough;
  const double skip_input_bytes = has_skip ? static_cast<double>(config.k) * sizeof(Ort::Float16_t) : 0.0;
  const double norm_scale_bytes = static_cast<double>(config.k) * sizeof(Ort::Float16_t);
  const double intermediate_bytes = is_unfused ? 4.0 * static_cast<double>(config.n) * sizeof(Ort::Float16_t) : 0.0;
  const double input_bytes = input_reads * static_cast<double>(config.k) * sizeof(Ort::Float16_t);
  const double packed_weight_bytes =
      2.0 * static_cast<double>(config.n) * static_cast<double>(k_blocks) * static_cast<double>(blob_size);
  const double scale_bytes = 2.0 * static_cast<double>(config.n) * static_cast<double>(k_blocks) * sizeof(Ort::Float16_t);
  const double output_bytes =
      static_cast<double>(config.n + (norm_kind == MlpNormKind::kSkipSimplifiedPassthrough ? config.k : 0)) *
      sizeof(Ort::Float16_t);

  return {
      input_bytes,
      packed_weight_bytes,
      scale_bytes,
      intermediate_bytes,
      output_bytes,
      input_bytes + skip_input_bytes + norm_scale_bytes + packed_weight_bytes + scale_bytes + intermediate_bytes + output_bytes,
  };
}

QkvTrafficStats CalculateQkvTrafficStats(const QkvDecodeBenchConfig& config,
                                         QkvDecodeBenchmarkVariant variant,
                                         QkvNormKind norm_kind) {
  const int64_t k_blocks = (config.k + config.block_size - 1) / config.block_size;
  const int64_t blob_size = (config.block_size * config.bits) / 8;

  const bool has_skip = norm_kind == QkvNormKind::kSkipSimplified ||
                        norm_kind == QkvNormKind::kSkipSimplifiedPassthrough;
  const bool has_skip_passthrough = norm_kind == QkvNormKind::kSkipSimplifiedPassthrough;

  const double input_bytes = static_cast<double>(config.k) * sizeof(Ort::Float16_t);
  const double skip_input_bytes = has_skip ? static_cast<double>(config.k) * sizeof(Ort::Float16_t) : 0.0;
  const double norm_scale_bytes = static_cast<double>(config.k) * sizeof(Ort::Float16_t);
  const double packed_weight_bytes =
      static_cast<double>(config.q_n + 2 * config.kv_n) * static_cast<double>(k_blocks) * static_cast<double>(blob_size);
  const double scale_bytes =
      static_cast<double>(config.q_n + 2 * config.kv_n) * static_cast<double>(k_blocks) * sizeof(Ort::Float16_t);
  const double intermediate_bytes =
      variant == QkvDecodeBenchmarkVariant::kUnfused ? static_cast<double>(config.k) * sizeof(Ort::Float16_t) : 0.0;
  const double output_bytes =
      static_cast<double>(config.q_n + 2 * config.kv_n + (has_skip_passthrough ? config.k : 0)) *
      sizeof(Ort::Float16_t);

  return {
      input_bytes,
      skip_input_bytes,
      norm_scale_bytes,
      packed_weight_bytes,
      scale_bytes,
      intermediate_bytes,
      output_bytes,
      input_bytes + skip_input_bytes + norm_scale_bytes + packed_weight_bytes + scale_bytes + intermediate_bytes + output_bytes,
  };
}

AdapterSelectionConfig GetAdapterSelectionConfig() {
  // Prefer a 5060 Ti when Dawn exposes one, otherwise fall back to the first
  // Dawn-enumerated adapter so the benchmark remains robust across machines.
  return {
      "5060 Ti",              // preferred_device_substring
      1,                      // context_id
      WGPUBackendType_D3D12,  // backend_type
      true,                   // print_adapter_list
  };
}

SelectedWebGpuContext CreateSelectedWebGpuContext() {
  const AdapterSelectionConfig config = GetAdapterSelectionConfig();

  wgpu::InstanceFeatureName required_instance_features[] = {wgpu::InstanceFeatureName::TimedWaitAny};
  wgpu::InstanceDescriptor instance_desc{};
  instance_desc.requiredFeatures = required_instance_features;
  instance_desc.requiredFeatureCount = sizeof(required_instance_features) / sizeof(required_instance_features[0]);

  SelectedWebGpuContext selected_context;
  selected_context.dawn_instance = std::make_unique<dawn::native::Instance>(&instance_desc);

#if !defined(BUILD_DAWN_SHARED_LIBRARY)
  static std::once_flag dawn_procs_initialized;
  std::call_once(dawn_procs_initialized, []() {
    dawnProcSetProcs(&dawn::native::GetProcs());
  });
#endif

  WGPURequestAdapterOptions adapter_options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
  adapter_options.backendType = config.backend_type;
  adapter_options.powerPreference = WGPUPowerPreference_Undefined;

  std::vector<dawn::native::Adapter> adapters = selected_context.dawn_instance->EnumerateAdapters(&adapter_options);
  if (adapters.empty()) {
    throw std::runtime_error("No Dawn adapters were found for the configured backend.");
  }

  std::vector<AdapterCandidate> candidates;
  candidates.reserve(adapters.size());
  int discrete_index = 0;
  int integrated_index = 0;
  int cpu_index = 0;
  int unknown_index = 0;
  for (size_t i = 0; i < adapters.size(); ++i) {
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(adapters[i].Get(), &info) != WGPUStatus_Success) {
      continue;
    }

    const WGPUAdapterType adapter_type = info.adapterType;
    int current_type_index = 0;
    switch (adapter_type) {
      case WGPUAdapterType_DiscreteGPU:
        current_type_index = discrete_index++;
        break;
      case WGPUAdapterType_IntegratedGPU:
        current_type_index = integrated_index++;
        break;
      case WGPUAdapterType_CPU:
        current_type_index = cpu_index++;
        break;
      default:
        current_type_index = unknown_index++;
        break;
    }
    candidates.push_back(AdapterCandidate{
        adapters[i],
        static_cast<int>(i),
        adapter_type,
        current_type_index,
        info.vendorID,
        info.deviceID,
        ToString(info.vendor),
        ToString(info.architecture),
        ToString(info.device),
        ToString(info.description),
    });

    wgpuAdapterInfoFreeMembers(info);
  }

  if (config.print_adapter_list) {
    std::cout << "Available Dawn GPU adapters for WebGPU benchmark:" << std::endl;
    bool printed_gpu = false;
    for (const auto& candidate : candidates) {
      if (!IsGpuAdapterType(candidate.adapter_type)) {
        continue;
      }

      printed_gpu = true;
      std::cout << "  " << FormatAdapterSummary(candidate)
                << " features={" << FormatFeatureSupport(candidate.adapter) << "}"
                << std::endl;
    }

    if (!printed_gpu) {
      std::cout << "  No integrated or discrete GPU adapters were found." << std::endl;
    }
  }

  AdapterCandidate* selected_adapter = nullptr;
  if (config.preferred_device_substring != nullptr) {
    const std::string preferred_substring = ToLower(config.preferred_device_substring);
    for (auto& candidate : candidates) {
      if (ToLower(candidate.device).find(preferred_substring) != std::string::npos) {
        selected_adapter = &candidate;
        break;
      }
    }
  }

  if (selected_adapter == nullptr && !candidates.empty()) {
    selected_adapter = &candidates.front();
  }

  if (selected_adapter == nullptr) {
    throw std::runtime_error("No Dawn adapter candidates were available for WebGPU benchmark selection.");
  }

  const wgpu::Adapter adapter = selected_adapter->adapter.Get();
  std::vector<wgpu::FeatureName> required_features = GetRequiredDeviceFeatures(adapter);
  wgpu::Limits required_limits = GetRequiredDeviceLimits(adapter);
  wgpu::DeviceDescriptor device_desc{};
  if (!required_features.empty()) {
    device_desc.requiredFeatures = required_features.data();
    device_desc.requiredFeatureCount = required_features.size();
  }
  device_desc.requiredLimits = &required_limits;

  selected_context.instance = selected_context.dawn_instance->Get();
  selected_context.device = selected_adapter->adapter.CreateDevice(&device_desc);
  if (selected_context.device == nullptr) {
    throw std::runtime_error("Failed to create a WGPUDevice for the selected adapter.");
  }

  selected_context.selected_adapter_summary = FormatAdapterSummary(*selected_adapter);
  std::cout << "Selected Dawn adapter for WebGPU benchmark: "
            << selected_context.selected_adapter_summary
            << " features={" << FormatFeatureSupport(selected_adapter->adapter) << "}"
            << std::endl;

  selected_context.provider_options["deviceId"] = std::to_string(config.context_id);
  selected_context.provider_options["webgpuInstance"] = std::to_string(reinterpret_cast<size_t>(selected_context.instance));
  selected_context.provider_options["webgpuDevice"] = std::to_string(reinterpret_cast<size_t>(selected_context.device));
  selected_context.provider_options["preserveDevice"] = "1";
  selected_context.provider_options["dawnProcTable"] = std::to_string(reinterpret_cast<size_t>(&dawn::native::GetProcs()));

  return selected_context;
}

const SelectedWebGpuContext& GetSelectedWebGpuContext() {
  static const SelectedWebGpuContext selected_context = CreateSelectedWebGpuContext();
  return selected_context;
}

template <typename T>
void AddTensorInitializer(ONNX_NAMESPACE::GraphProto& graph,
                          const std::string& name,
                          int32_t data_type,
                          const std::vector<int64_t>& dims,
                          const std::vector<T>& values) {
  auto* initializer = graph.add_initializer();
  initializer->set_name(name);
  initializer->set_data_type(data_type);
  for (int64_t dim : dims) {
    initializer->add_dims(dim);
  }

  initializer->set_raw_data(values.data(), values.size() * sizeof(T));
}

void AddTensorValueInfo(ONNX_NAMESPACE::GraphProto& graph,
                        const std::string& name,
                        int32_t data_type,
                        const std::vector<int64_t>& dims) {
  auto* value_info = graph.add_value_info();
  value_info->set_name(name);
  value_info->mutable_type()->mutable_tensor_type()->set_elem_type(data_type);
  auto* shape = value_info->mutable_type()->mutable_tensor_type()->mutable_shape();
  for (int64_t dim : dims) {
    shape->add_dim()->set_dim_value(dim);
  }
}

std::vector<MlpDecodeBenchConfig> GetMlpDecodeBenchConfigs() {
  // Qwen3-1.7B MLP gate/up decode geometry: hidden=2048, intermediate=6144.
  return {
      {6144, 2048, 4, 32, 4},
  };
}

std::vector<QkvDecodeBenchConfig> GetQkvDecodeBenchConfigs() {
  // Qwen3-1.7B attention projection geometry: hidden=2048, q=2048, kv=1024.
  return {
      {2048, 1024, 2048, 4, 32, 4},
  };
}

void AddMatMulNBitsNode(ONNX_NAMESPACE::GraphProto& graph,
                        const std::string& node_name,
                        const std::string& input_name,
                        const std::string& weight_name,
                        const std::string& scale_name,
                        const std::string& output_name,
                        int64_t k,
                        int64_t n,
                        int64_t bits,
                        int64_t block_size,
                        int64_t accuracy_level) {
  auto* node = graph.add_node();
  node->set_name(node_name);
  node->set_op_type("MatMulNBits");
  node->set_domain("com.microsoft");
  node->add_input(input_name);
  node->add_input(weight_name);
  node->add_input(scale_name);
  node->add_input("");
  node->add_input("");
  node->add_output(output_name);

  auto* attr_k = node->add_attribute();
  attr_k->set_name("K");
  attr_k->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_k->set_i(k);

  auto* attr_n = node->add_attribute();
  attr_n->set_name("N");
  attr_n->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_n->set_i(n);

  auto* attr_bits = node->add_attribute();
  attr_bits->set_name("bits");
  attr_bits->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_bits->set_i(bits);

  auto* attr_block = node->add_attribute();
  attr_block->set_name("block_size");
  attr_block->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_block->set_i(block_size);

  auto* attr_accuracy = node->add_attribute();
  attr_accuracy->set_name("accuracy_level");
  attr_accuracy->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_accuracy->set_i(accuracy_level);
}

void AddMatMulNBitsMlpNode(ONNX_NAMESPACE::GraphProto& graph,
                           const std::string& node_name,
                           const std::string& input_name,
                           const std::string& skip_input_name,
                           const std::string& norm_scale_name,
                           const std::string& gate_weight_name,
                           const std::string& gate_scale_name,
                           const std::string& up_weight_name,
                           const std::string& up_scale_name,
                           const std::string& output_name,
                           const std::string& skip_sum_output_name,
                           int64_t k,
                           int64_t n,
                           int64_t bits,
                           int64_t block_size,
                           int64_t accuracy_level) {
  auto* node = graph.add_node();
  node->set_name(node_name);
  node->set_op_type("MatMulNBitsMlp");
  node->set_domain("com.microsoft");
  node->add_input(input_name);
  node->add_input(skip_input_name);
  node->add_input(norm_scale_name);
  node->add_input(gate_weight_name);
  node->add_input(gate_scale_name);
  node->add_input("");
  node->add_input(up_weight_name);
  node->add_input(up_scale_name);
  node->add_input("");
  node->add_output(output_name);
  if (!skip_sum_output_name.empty()) {
    node->add_output(skip_sum_output_name);
  }

  auto* attr_k = node->add_attribute();
  attr_k->set_name("K");
  attr_k->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_k->set_i(k);

  auto* attr_n = node->add_attribute();
  attr_n->set_name("N");
  attr_n->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_n->set_i(n);

  auto* attr_bits = node->add_attribute();
  attr_bits->set_name("bits");
  attr_bits->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_bits->set_i(bits);

  auto* attr_block = node->add_attribute();
  attr_block->set_name("block_size");
  attr_block->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_block->set_i(block_size);

  auto* attr_accuracy = node->add_attribute();
  attr_accuracy->set_name("accuracy_level");
  attr_accuracy->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_accuracy->set_i(accuracy_level);

  auto* attr_activation = node->add_attribute();
  attr_activation->set_name("activation");
  attr_activation->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr_activation->set_s("silu");
}

void AddMatMulNBitsQkvNode(ONNX_NAMESPACE::GraphProto& graph,
                           const std::string& node_name,
                           const std::string& input_name,
                           const std::string& skip_input_name,
                           const std::string& norm_scale_name,
                           const std::string& q_weight_name,
                           const std::string& q_scale_name,
                           const std::string& k_weight_name,
                           const std::string& k_scale_name,
                           const std::string& v_weight_name,
                           const std::string& v_scale_name,
                           const std::string& q_output_name,
                           const std::string& k_output_name,
                           const std::string& v_output_name,
                           const std::string& skip_sum_output_name,
                           int64_t k,
                           int64_t q_n,
                           int64_t kv_n,
                           int64_t bits,
                           int64_t block_size,
                           int64_t accuracy_level,
                           float epsilon) {
  auto* node = graph.add_node();
  node->set_name(node_name);
  node->set_op_type("MatMulNBitsQkv");
  node->set_domain("com.microsoft");
  node->add_input(input_name);
  node->add_input(skip_input_name);
  node->add_input(norm_scale_name);
  node->add_input(q_weight_name);
  node->add_input(q_scale_name);
  node->add_input(k_weight_name);
  node->add_input(k_scale_name);
  node->add_input(v_weight_name);
  node->add_input(v_scale_name);
  node->add_output(q_output_name);
  node->add_output(k_output_name);
  node->add_output(v_output_name);
  if (!skip_sum_output_name.empty()) {
    node->add_output(skip_sum_output_name);
  }

  auto* attr_k = node->add_attribute();
  attr_k->set_name("K");
  attr_k->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_k->set_i(k);
  auto* attr_qn = node->add_attribute();
  attr_qn->set_name("Nq");
  attr_qn->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_qn->set_i(q_n);
  auto* attr_kvn = node->add_attribute();
  attr_kvn->set_name("Nkv");
  attr_kvn->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_kvn->set_i(kv_n);
  auto* attr_bits = node->add_attribute();
  attr_bits->set_name("bits");
  attr_bits->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_bits->set_i(bits);
  auto* attr_block = node->add_attribute();
  attr_block->set_name("block_size");
  attr_block->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_block->set_i(block_size);
  auto* attr_accuracy = node->add_attribute();
  attr_accuracy->set_name("accuracy_level");
  attr_accuracy->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_accuracy->set_i(accuracy_level);
  auto* attr_epsilon = node->add_attribute();
  attr_epsilon->set_name("epsilon");
  attr_epsilon->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attr_epsilon->set_f(epsilon);
}

std::string GetMlpVariantLabel(MlpDecodeBenchmarkVariant variant) {
  switch (variant) {
    case MlpDecodeBenchmarkVariant::kUnfused:
      return "unfused";
    case MlpDecodeBenchmarkVariant::kFused:
      return "fused";
  }

  return "unknown";
}

std::string GetMlpNormKindLabel(MlpNormKind norm_kind) {
  switch (norm_kind) {
    case MlpNormKind::kSimplified:
      return "simplified";
    case MlpNormKind::kSkipSimplified:
      return "skip_simplified";
    case MlpNormKind::kSkipSimplifiedPassthrough:
      return "skip_simplified_passthrough";
  }

  return "unknown";
}

std::string GetMlpDecodeBenchmarkLabel(MlpDecodeBenchmarkVariant variant, MlpNormKind norm_kind) {
  std::ostringstream stream;
  stream << "fp16_mlp_decode_" << GetMlpNormKindLabel(norm_kind) << '_' << GetMlpVariantLabel(variant) << '_'
         << (IsDecodeBenchmarkPerfMode() ? "perf" : "correctness") << '_'
         << "auto_gpu_"
         << (IsGraphCaptureBenchmarkEnabled() ? "graph_on" : "graph_off");
  return stream.str();
}

std::string GetQkvVariantLabel(QkvDecodeBenchmarkVariant variant) {
  return variant == QkvDecodeBenchmarkVariant::kFused ? "fused" : "unfused";
}

std::string GetQkvNormKindLabel(QkvNormKind norm_kind) {
  switch (norm_kind) {
    case QkvNormKind::kSimplified:
      return "simplified";
    case QkvNormKind::kSkipSimplified:
      return "skip_simplified";
    case QkvNormKind::kSkipSimplifiedPassthrough:
      return "skip_simplified_passthrough";
  }

  return "unknown";
}

std::string GetQkvDecodeBenchmarkLabel(QkvDecodeBenchmarkVariant variant, QkvNormKind norm_kind) {
  std::ostringstream stream;
  stream << "fp16_qkv_norm_" << GetQkvNormKindLabel(norm_kind) << '_' << GetQkvVariantLabel(variant) << '_'
         << (IsDecodeBenchmarkPerfMode() ? "perf" : "correctness") << '_'
         << "auto_gpu_"
         << (IsGraphCaptureBenchmarkEnabled() ? "graph_on" : "graph_off");
  return stream.str();
}

std::vector<uint8_t> SerializeMatMulNBitsMlpModel(const MlpDecodeBenchConfig& config,
                                                  MlpDecodeBenchmarkVariant variant,
                                                  MlpNormKind norm_kind) {
  const int64_t k_blocks = (config.k + config.block_size - 1) / config.block_size;
  const int64_t blob_size = (config.block_size * config.bits) / 8;

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(10);

  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(21);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  switch (variant) {
    case MlpDecodeBenchmarkVariant::kFused:
      graph->set_name("WebGpuMatMulNBitsMlpDecodeFused");
      break;
    case MlpDecodeBenchmarkVariant::kUnfused:
    default:
      graph->set_name("WebGpuMatMulNBitsMlpDecodeUnfused");
      break;
  }

  const bool has_skip = norm_kind == MlpNormKind::kSkipSimplified ||
                        norm_kind == MlpNormKind::kSkipSimplifiedPassthrough;
  const bool has_skip_passthrough = norm_kind == MlpNormKind::kSkipSimplifiedPassthrough;

  auto* input = graph->add_input();
  input->set_name("A");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.k);

  if (has_skip) {
    auto* skip_input = graph->add_input();
    skip_input->set_name("Skip");
    skip_input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    skip_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    skip_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.k);
  }

  auto* output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.n);
  if (has_skip_passthrough) {
    auto* skip_sum_output = graph->add_output();
    skip_sum_output->set_name("SkipSum");
    skip_sum_output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    skip_sum_output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    skip_sum_output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.k);
  }

  std::vector<uint8_t> gate_b(static_cast<size_t>(config.n * k_blocks * blob_size), uint8_t{0x11});
  std::vector<uint8_t> up_b(static_cast<size_t>(config.n * k_blocks * blob_size), uint8_t{0x77});
  std::vector<Ort::Float16_t> gate_scales(static_cast<size_t>(config.n * k_blocks), Ort::Float16_t(0.03125f));
  std::vector<Ort::Float16_t> up_scales(static_cast<size_t>(config.n * k_blocks), Ort::Float16_t(0.0625f));
  std::vector<Ort::Float16_t> norm_scale(static_cast<size_t>(config.k), Ort::Float16_t(1.0f));
  AddTensorInitializer(*graph, "gate_B", ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                       {config.n, k_blocks, blob_size}, gate_b);
  AddTensorInitializer(*graph, "up_B", ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                       {config.n, k_blocks, blob_size}, up_b);
  AddTensorInitializer(*graph, "gate_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                       {config.n, k_blocks}, gate_scales);
  AddTensorInitializer(*graph, "up_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                       {config.n, k_blocks}, up_scales);
  AddTensorInitializer(*graph, "norm_scale", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                       {config.k}, norm_scale);

  if (variant == MlpDecodeBenchmarkVariant::kFused) {
    AddMatMulNBitsMlpNode(*graph,
                          "MatMulNBitsMlpDecode",
                          "A",
                          has_skip ? "Skip" : "",
                          "norm_scale",
                          "gate_B",
                          "gate_scales",
                          "up_B",
                          "up_scales",
                          "Y",
                          has_skip_passthrough ? "SkipSum" : "",
                          config.k,
                          config.n,
                          config.bits,
                          config.block_size,
                          config.accuracy_level);
  } else {
    const char* mlp_input_name = "A_norm";
    AddTensorValueInfo(*graph, "A_norm", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.k});
    auto* norm = graph->add_node();
    norm->set_name(has_skip ? "InputSkipSimplifiedLayerNorm" : "InputSimplifiedLayerNorm");
    norm->set_op_type(has_skip ? "SkipSimplifiedLayerNormalization" : "SimplifiedLayerNormalization");
    if (has_skip) {
      norm->set_domain("com.microsoft");
      norm->add_input("A");
      norm->add_input("Skip");
      norm->add_input("norm_scale");
      norm->add_output("A_norm");
      if (has_skip_passthrough) {
        norm->add_output("");
        norm->add_output("");
        norm->add_output("SkipSum");
      }
    } else {
      norm->add_input("A");
      norm->add_input("norm_scale");
      norm->add_output("A_norm");
    }
    auto* attr_epsilon = norm->add_attribute();
    attr_epsilon->set_name("epsilon");
    attr_epsilon->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attr_epsilon->set_f(1e-6f);

    AddTensorValueInfo(*graph, "gate_mm", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.n});
    AddTensorValueInfo(*graph, "up_mm", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.n});
    AddTensorValueInfo(*graph, "gate_sigmoid", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.n});
    AddTensorValueInfo(*graph, "gate_silu", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.n});

    AddMatMulNBitsNode(*graph,
                       "GateMatMulNBitsDecode",
                       mlp_input_name,
                       "gate_B",
                       "gate_scales",
                       "gate_mm",
                       config.k,
                       config.n,
                       config.bits,
                       config.block_size,
                       config.accuracy_level);
    AddMatMulNBitsNode(*graph,
                       "UpMatMulNBitsDecode",
                       mlp_input_name,
                       "up_B",
                       "up_scales",
                       "up_mm",
                       config.k,
                       config.n,
                       config.bits,
                       config.block_size,
                       config.accuracy_level);

    auto* sigmoid = graph->add_node();
    sigmoid->set_name("GateSigmoid");
    sigmoid->set_op_type("Sigmoid");
    sigmoid->add_input("gate_mm");
    sigmoid->add_output("gate_sigmoid");

    auto* silu_mul = graph->add_node();
    silu_mul->set_name("GateSiluMul");
    silu_mul->set_op_type("Mul");
    silu_mul->add_input("gate_mm");
    silu_mul->add_input("gate_sigmoid");
    silu_mul->add_output("gate_silu");

    auto* output_mul = graph->add_node();
    output_mul->set_name("GateUpMul");
    output_mul->set_op_type("Mul");
    output_mul->add_input("gate_silu");
    output_mul->add_input("up_mm");
    output_mul->add_output("Y");
  }

  const auto serialized = model.SerializeAsString();
  return std::vector<uint8_t>(serialized.begin(), serialized.end());
}

std::vector<uint8_t> SerializeMatMulNBitsQkvModel(const QkvDecodeBenchConfig& config,
                                                  QkvDecodeBenchmarkVariant variant,
                                                  QkvNormKind norm_kind) {
  const int64_t k_blocks = (config.k + config.block_size - 1) / config.block_size;
  const int64_t blob_size = (config.block_size * config.bits) / 8;

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(10);

  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(21);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name(variant == QkvDecodeBenchmarkVariant::kFused
                      ? (norm_kind == QkvNormKind::kSkipSimplified ? "WebGpuMatMulNBitsQkvSkipNormFused" : "WebGpuMatMulNBitsQkvSimplifiedNormFused")
                      : (norm_kind == QkvNormKind::kSkipSimplified ? "WebGpuMatMulNBitsQkvSkipNormUnfused" : "WebGpuMatMulNBitsQkvSimplifiedNormUnfused"));

  const bool has_skip = norm_kind == QkvNormKind::kSkipSimplified ||
                        norm_kind == QkvNormKind::kSkipSimplifiedPassthrough;
  const bool has_skip_passthrough = norm_kind == QkvNormKind::kSkipSimplifiedPassthrough;

  auto* input = graph->add_input();
  input->set_name("A");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.k);

  if (has_skip) {
    auto* skip_input = graph->add_input();
    skip_input->set_name("Skip");
    skip_input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    skip_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    skip_input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.k);
  }

  auto add_output = [&](const std::string& name, int64_t n) {
    auto* output = graph->add_output();
    output->set_name(name);
    output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(n);
  };
  add_output("Q", config.q_n);
  add_output("K", config.kv_n);
  add_output("V", config.kv_n);
  if (has_skip_passthrough) {
    add_output("SkipSum", config.k);
  }

  std::vector<Ort::Float16_t> norm_scale(static_cast<size_t>(config.k), Ort::Float16_t(1.0f));
  std::vector<uint8_t> q_b(static_cast<size_t>(config.q_n * k_blocks * blob_size), uint8_t{0x11});
  std::vector<uint8_t> k_b(static_cast<size_t>(config.kv_n * k_blocks * blob_size), uint8_t{0x33});
  std::vector<uint8_t> v_b(static_cast<size_t>(config.kv_n * k_blocks * blob_size), uint8_t{0x77});
  std::vector<Ort::Float16_t> q_scales(static_cast<size_t>(config.q_n * k_blocks), Ort::Float16_t(0.03125f));
  std::vector<Ort::Float16_t> k_scales(static_cast<size_t>(config.kv_n * k_blocks), Ort::Float16_t(0.03125f));
  std::vector<Ort::Float16_t> v_scales(static_cast<size_t>(config.kv_n * k_blocks), Ort::Float16_t(0.0625f));

  AddTensorInitializer(*graph, "norm_scale", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {config.k}, norm_scale);
  AddTensorInitializer(*graph, "q_B", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {config.q_n, k_blocks, blob_size}, q_b);
  AddTensorInitializer(*graph, "q_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {config.q_n, k_blocks}, q_scales);
  AddTensorInitializer(*graph, "k_B", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {config.kv_n, k_blocks, blob_size}, k_b);
  AddTensorInitializer(*graph, "k_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {config.kv_n, k_blocks}, k_scales);
  AddTensorInitializer(*graph, "v_B", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {config.kv_n, k_blocks, blob_size}, v_b);
  AddTensorInitializer(*graph, "v_scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {config.kv_n, k_blocks}, v_scales);

  if (variant == QkvDecodeBenchmarkVariant::kFused) {
    AddMatMulNBitsQkvNode(*graph,
                          "MatMulNBitsQkvDecode",
                          "A",
                          has_skip ? "Skip" : "",
                          "norm_scale",
                          "q_B",
                          "q_scales",
                          "k_B",
                          "k_scales",
                          "v_B",
                          "v_scales",
                          "Q",
                          "K",
                          "V",
                          has_skip_passthrough ? "SkipSum" : "",
                          config.k,
                          config.q_n,
                          config.kv_n,
                          config.bits,
                          config.block_size,
                          config.accuracy_level,
                          1e-6f);
  } else {
    AddTensorValueInfo(*graph, "A_norm", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.k});
    auto* norm = graph->add_node();
    norm->set_name(has_skip ? "InputSkipSimplifiedLayerNorm" : "InputSimplifiedLayerNorm");
    norm->set_op_type(has_skip ? "SkipSimplifiedLayerNormalization" : "SimplifiedLayerNormalization");
    if (has_skip) {
      if (has_skip_passthrough) {
        AddTensorValueInfo(*graph, "SkipSum", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, {1, config.k});
      }
      norm->set_domain("com.microsoft");
      norm->add_input("A");
      norm->add_input("Skip");
      norm->add_input("norm_scale");
      norm->add_output("A_norm");
      if (has_skip_passthrough) {
        norm->add_output("");
        norm->add_output("");
        norm->add_output("SkipSum");
      }
    } else {
      norm->add_input("A");
      norm->add_input("norm_scale");
      norm->add_output("A_norm");
    }
    auto* attr_epsilon = norm->add_attribute();
    attr_epsilon->set_name("epsilon");
    attr_epsilon->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attr_epsilon->set_f(1e-6f);

    AddMatMulNBitsNode(*graph, "QMatMulNBitsDecode", "A_norm", "q_B", "q_scales", "Q", config.k, config.q_n, config.bits, config.block_size, config.accuracy_level);
    AddMatMulNBitsNode(*graph, "KMatMulNBitsDecode", "A_norm", "k_B", "k_scales", "K", config.k, config.kv_n, config.bits, config.block_size, config.accuracy_level);
    AddMatMulNBitsNode(*graph, "VMatMulNBitsDecode", "A_norm", "v_B", "v_scales", "V", config.k, config.kv_n, config.bits, config.block_size, config.accuracy_level);
  }

  const auto serialized = model.SerializeAsString();
  return std::vector<uint8_t>(serialized.begin(), serialized.end());
}

Ort::Session CreateSessionFromModelData(const std::vector<uint8_t>& model_data,
                                        const std::unordered_map<std::string, std::string>* provider_options,
                                        GraphOptimizationLevel graph_optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL) {
  Ort::SessionOptions session_options;
  session_options.DisableMemPattern();
  session_options.SetGraphOptimizationLevel(graph_optimization_level);
  if (IsVerboseSessionLogEnabled()) {
    session_options.SetLogSeverityLevel(0);
  }

  const std::string optimized_model_path = GetOptimizedModelPath();
  if (!optimized_model_path.empty()) {
    const auto optimized_model_path_ort = onnxruntime::ToWideString(optimized_model_path);
    session_options.SetOptimizedModelFilePath(optimized_model_path_ort.c_str());
  }

  if (provider_options != nullptr) {
    if (IsGraphCaptureBenchmarkEnabled()) {
      session_options.AddConfigEntry(onnxruntime::webgpu::options::kEnableGraphCapture,
                                     onnxruntime::webgpu::options::kEnableGraphCapture_ON);
    }
    session_options.AppendExecutionProvider("WebGPU", *provider_options);
  }

  OrtSession* raw_session = nullptr;
  OrtStatus* status = g_ort->CreateSessionFromArray(env, model_data.data(), model_data.size(), session_options, &raw_session);
  if (status != nullptr) {
    std::string error_message = g_ort->GetErrorMessage(status);
    g_ort->ReleaseStatus(status);
    throw std::runtime_error(error_message);
  }

  return Ort::Session{raw_session};
}

void ValidateDecodeOutputs(const std::vector<uint8_t>& model_data,
                           Ort::Session& webgpu_session,
                           const char* const* input_names,
                           const Ort::Value* input_tensor,
                           const char* const* output_names) {
  Ort::Session cpu_session = CreateSessionFromModelData(model_data, nullptr);

  auto webgpu_outputs = webgpu_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensor, 1, output_names, 1);
  auto cpu_outputs = cpu_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensor, 1, output_names, 1);

  if (webgpu_outputs.size() != 1 || cpu_outputs.size() != 1) {
    throw std::runtime_error("Expected a single output from both WebGPU and CPU sessions.");
  }

  const auto& webgpu_output = webgpu_outputs[0];
  const auto& cpu_output = cpu_outputs[0];
  const size_t element_count = webgpu_output.GetTensorTypeAndShapeInfo().GetElementCount();
  if (element_count != cpu_output.GetTensorTypeAndShapeInfo().GetElementCount()) {
    throw std::runtime_error("WebGPU and CPU output sizes do not match.");
  }

  const auto* webgpu_data = webgpu_output.GetTensorData<Ort::Float16_t>();
  const auto* cpu_data = cpu_output.GetTensorData<Ort::Float16_t>();
  float max_abs_diff = 0.0f;
  size_t max_abs_diff_index = 0;
  for (size_t i = 0; i < element_count; ++i) {
    const float webgpu_value = webgpu_data[i].ToFloat();
    const float cpu_value = cpu_data[i].ToFloat();
    const float abs_diff = std::abs(webgpu_value - cpu_value);
    const float allowed_diff = kDecodeCorrectnessAbsTolerance +
                               kDecodeCorrectnessRelTolerance * std::max(std::abs(webgpu_value), std::abs(cpu_value));
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
      max_abs_diff_index = i;
    }
    if (abs_diff > allowed_diff) {
      std::ostringstream stream;
      stream << "Decode correctness check failed at index " << i
             << ": webgpu=" << webgpu_value
             << ", cpu=" << cpu_value
             << ", abs_diff=" << abs_diff
             << ", allowed_diff=" << allowed_diff;
      throw std::runtime_error(stream.str());
    }
  }

  std::cout << "Decode correctness check passed. max_abs_diff=" << max_abs_diff
            << " at index " << max_abs_diff_index << std::endl;
}

void ValidateMlpDecodeOutputs(const std::vector<uint8_t>& unfused_model_data,
                              const std::vector<uint8_t>& fused_model_data,
                              const std::unordered_map<std::string, std::string>& provider_options,
                              const char* const* input_names,
                              const Ort::Value* input_tensors,
                              size_t input_count,
                              const char* const* output_names,
                              size_t output_count) {
  Ort::Session unfused_session = CreateSessionFromModelData(unfused_model_data,
                                                            &provider_options,
                                                            GraphOptimizationLevel::ORT_DISABLE_ALL);
  Ort::Session fused_session = CreateSessionFromModelData(fused_model_data,
                                                          &provider_options,
                                                          GraphOptimizationLevel::ORT_ENABLE_ALL);

  auto unfused_outputs = unfused_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, input_count, output_names, output_count);
  auto fused_outputs = fused_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, input_count, output_names, output_count);

  for (size_t output_index = 0; output_index < output_count; ++output_index) {
    const auto& unfused_output = unfused_outputs[output_index];
    const auto& fused_output = fused_outputs[output_index];
    const size_t element_count = unfused_output.GetTensorTypeAndShapeInfo().GetElementCount();
    if (element_count != fused_output.GetTensorTypeAndShapeInfo().GetElementCount()) {
      throw std::runtime_error("Unfused and fused MLP output sizes do not match.");
    }

    const auto* unfused_data = unfused_output.GetTensorData<Ort::Float16_t>();
    const auto* fused_data = fused_output.GetTensorData<Ort::Float16_t>();
    float max_abs_diff = 0.0f;
    size_t max_abs_diff_index = 0;
    for (size_t i = 0; i < element_count; ++i) {
      const float unfused_value = unfused_data[i].ToFloat();
      const float fused_value = fused_data[i].ToFloat();
      const float abs_diff = std::abs(unfused_value - fused_value);
      const float allowed_diff = kDecodeCorrectnessAbsTolerance +
                                 kDecodeCorrectnessRelTolerance * std::max(std::abs(unfused_value), std::abs(fused_value));
      if (abs_diff > max_abs_diff) {
        max_abs_diff = abs_diff;
        max_abs_diff_index = i;
      }
      if (abs_diff > allowed_diff) {
        std::ostringstream stream;
        stream << "MLP decode correctness check failed on output " << output_index
               << " at index " << i
               << ": unfused=" << unfused_value
               << ", fused=" << fused_value
               << ", abs_diff=" << abs_diff
               << ", allowed_diff=" << allowed_diff;
        throw std::runtime_error(stream.str());
      }
    }

    std::cout << "MLP decode correctness check passed for output " << output_index
              << ". max_abs_diff=" << max_abs_diff
              << " at index " << max_abs_diff_index << std::endl;
  }
}

void ValidateQkvDecodeOutputs(const std::vector<uint8_t>& unfused_model_data,
                              const std::vector<uint8_t>& fused_model_data,
                              const std::unordered_map<std::string, std::string>& provider_options,
                              const char* const* input_names,
                              const Ort::Value* input_tensors,
                              size_t input_count,
                              const char* const* output_names,
                              size_t output_count) {
  Ort::Session unfused_session = CreateSessionFromModelData(unfused_model_data, &provider_options, GraphOptimizationLevel::ORT_DISABLE_ALL);
  Ort::Session fused_session = CreateSessionFromModelData(fused_model_data, &provider_options, GraphOptimizationLevel::ORT_DISABLE_ALL);

  auto unfused_outputs = unfused_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, input_count, output_names, output_count);
  auto fused_outputs = fused_session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, input_count, output_names, output_count);

  for (size_t output_index = 0; output_index < output_count; ++output_index) {
    const size_t element_count = unfused_outputs[output_index].GetTensorTypeAndShapeInfo().GetElementCount();
    const auto* unfused_data = unfused_outputs[output_index].GetTensorData<Ort::Float16_t>();
    const auto* fused_data = fused_outputs[output_index].GetTensorData<Ort::Float16_t>();
    for (size_t i = 0; i < element_count; ++i) {
      const float unfused_value = unfused_data[i].ToFloat();
      const float fused_value = fused_data[i].ToFloat();
      const float abs_diff = std::abs(unfused_value - fused_value);
      const float allowed_diff = kDecodeCorrectnessAbsTolerance +
                                 kDecodeCorrectnessRelTolerance * std::max(std::abs(unfused_value), std::abs(fused_value));
      if (abs_diff > allowed_diff) {
        std::ostringstream stream;
        stream << "QKV decode correctness check failed on output " << output_index
               << " at index " << i
               << ": unfused=" << unfused_value
               << ", fused=" << fused_value
               << ", abs_diff=" << abs_diff
               << ", allowed_diff=" << allowed_diff;
        throw std::runtime_error(stream.str());
      }
    }
  }

  std::cout << "QKV decode correctness check passed." << std::endl;
}

void BenchmarkWebGpuMatMulNBitsQkvDecode(benchmark::State& state,
                                         QkvDecodeBenchmarkVariant variant,
                                         QkvNormKind norm_kind) {
  try {
    const QkvDecodeBenchConfig config{
        state.range(0),
        state.range(1),
        state.range(2),
        state.range(3),
        state.range(4),
        state.range(5),
    };

    if (config.k % config.block_size != 0) {
      state.SkipWithError("K must be divisible by block_size for this benchmark skeleton.");
      return;
    }

    const QkvTrafficStats traffic = CalculateQkvTrafficStats(config, variant, norm_kind);
    std::vector<uint8_t> model_data = SerializeMatMulNBitsQkvModel(config, variant, norm_kind);
    const SelectedWebGpuContext& selected_context = GetSelectedWebGpuContext();
    Ort::Session session = CreateSessionFromModelData(model_data,
                                                      &selected_context.provider_options,
                                                      GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape{1, config.k};
    std::vector<Ort::Float16_t> activation(static_cast<size_t>(config.k));
    std::vector<Ort::Float16_t> skip_activation(static_cast<size_t>(config.k));

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : activation) {
      value = Ort::Float16_t(dist(rng));
    }
    for (auto& value : skip_activation) {
      value = Ort::Float16_t(dist(rng));
    }

    const bool has_skip = norm_kind == QkvNormKind::kSkipSimplified ||
                          norm_kind == QkvNormKind::kSkipSimplifiedPassthrough;
    const bool has_skip_passthrough = norm_kind == QkvNormKind::kSkipSimplifiedPassthrough;
    const char* simplified_input_names[] = {"A"};
    const char* skip_input_names[] = {"A", "Skip"};
    const char* simplified_output_names[] = {"Q", "K", "V"};
    const char* skip_output_names[] = {"Q", "K", "V"};
    const char* skip_passthrough_output_names[] = {"Q", "K", "V", "SkipSum"};
    const char* const* input_names = has_skip ? skip_input_names : simplified_input_names;
    const char* const* output_names = has_skip_passthrough ? skip_passthrough_output_names
                                                           : (has_skip ? skip_output_names : simplified_output_names);
    const size_t input_count = has_skip ? 2u : 1u;
    const size_t output_count = has_skip_passthrough ? 4u : 3u;

    std::array<Ort::Value, 2> input_tensors = {
        Ort::Value::CreateTensor<Ort::Float16_t>(memory_info,
                                                 activation.data(),
                                                 activation.size(),
                                                 input_shape.data(),
                                                 input_shape.size()),
        Ort::Value::CreateTensor<Ort::Float16_t>(memory_info,
                                                 skip_activation.data(),
                                                 skip_activation.size(),
                                                 input_shape.data(),
                                                 input_shape.size())};
    Ort::RunOptions run_options = CreateBenchmarkRunOptions();

    if (!IsDecodeBenchmarkPerfMode() && variant == QkvDecodeBenchmarkVariant::kFused) {
      ValidateQkvDecodeOutputs(SerializeMatMulNBitsQkvModel(config, QkvDecodeBenchmarkVariant::kUnfused, norm_kind),
                               model_data,
                               selected_context.provider_options,
                               input_names,
                               input_tensors.data(),
                               input_count,
                               output_names,
                               output_count);
    }

    for (int i = 0; i < kDecodeWarmupRuns; ++i) {
      auto warmup_outputs = session.Run(run_options, input_names, input_tensors.data(), input_count, output_names, output_count);
      benchmark::DoNotOptimize(warmup_outputs);
    }

    double total_kernel_seconds = 0.0;
    for (auto _ : state) {
      const auto kernel_start = std::chrono::steady_clock::now();
      auto outputs = session.Run(run_options, input_names, input_tensors.data(), input_count, output_names, output_count);
      const auto kernel_end = std::chrono::steady_clock::now();
      total_kernel_seconds += std::chrono::duration<double>(kernel_end - kernel_start).count();
      benchmark::DoNotOptimize(outputs);
    }

    const double total_flops = 2.0 * static_cast<double>(config.k) * static_cast<double>(config.q_n + 2 * config.kv_n);
    const double achieved_bandwidth_bytes_per_second =
        total_kernel_seconds > 0.0
            ? traffic.total_bytes * static_cast<double>(state.iterations()) / total_kernel_seconds
            : 0.0;

    state.SetLabel(GetQkvDecodeBenchmarkLabel(variant, norm_kind));
    state.counters["TFLOPS"] = benchmark::Counter(total_flops, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["ApproxMemBW_GBps"] = benchmark::Counter(achieved_bandwidth_bytes_per_second / 1.0e9);
    state.counters["ApproxTraffic_MB"] = benchmark::Counter(traffic.total_bytes / 1.0e6);
    state.counters["Input_MB"] = benchmark::Counter(traffic.input_bytes / 1.0e6);
    state.counters["SkipInput_MB"] = benchmark::Counter(traffic.skip_input_bytes / 1.0e6);
    state.counters["NormScale_MB"] = benchmark::Counter(traffic.norm_scale_bytes / 1.0e6);
    state.counters["PackedW_MB"] = benchmark::Counter(traffic.packed_weight_bytes / 1.0e6);
    state.counters["Scales_MB"] = benchmark::Counter(traffic.scale_bytes / 1.0e6);
    state.counters["Intermediate_MB"] = benchmark::Counter(traffic.intermediate_bytes / 1.0e6);
    state.counters["Output_MB"] = benchmark::Counter(traffic.output_bytes / 1.0e6);
    state.counters["GraphReplay"] = benchmark::Counter(IsGraphCaptureBenchmarkEnabled() ? 1.0 : 0.0);
  } catch (const std::exception& ex) {
    state.SkipWithError(ex.what());
  }
}

void BenchmarkWebGpuMatMulNBitsMlpDecode(benchmark::State& state,
                                         MlpDecodeBenchmarkVariant variant,
                                         MlpNormKind norm_kind) {
  try {
    const MlpDecodeBenchConfig config{
        state.range(0),
        state.range(1),
        state.range(2),
        state.range(3),
        state.range(4),
    };

    if (config.k % config.block_size != 0) {
      state.SkipWithError("K must be divisible by block_size for this benchmark skeleton.");
      return;
    }

    const MlpTrafficStats traffic = CalculateMlpTrafficStats(config, variant, norm_kind);
    std::vector<uint8_t> model_data = SerializeMatMulNBitsMlpModel(config, variant, norm_kind);
    const SelectedWebGpuContext& selected_context = GetSelectedWebGpuContext();
    const GraphOptimizationLevel optimization_level =
        variant == MlpDecodeBenchmarkVariant::kUnfused ? GraphOptimizationLevel::ORT_DISABLE_ALL
                                                       : GraphOptimizationLevel::ORT_ENABLE_ALL;
    Ort::Session session = CreateSessionFromModelData(model_data,
                                                      &selected_context.provider_options,
                                                      optimization_level);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape{1, config.k};
    std::vector<Ort::Float16_t> activation(static_cast<size_t>(config.k));
    std::vector<Ort::Float16_t> skip_activation(static_cast<size_t>(config.k));
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : activation) {
      value = Ort::Float16_t(dist(rng));
    }
    for (auto& value : skip_activation) {
      value = Ort::Float16_t(dist(rng));
    }

    const bool has_skip = norm_kind == MlpNormKind::kSkipSimplified ||
                          norm_kind == MlpNormKind::kSkipSimplifiedPassthrough;
    const bool has_skip_passthrough = norm_kind == MlpNormKind::kSkipSimplifiedPassthrough;
    const char* simplified_input_names[] = {"A"};
    const char* skip_input_names[] = {"A", "Skip"};
    const char* main_output_names[] = {"Y"};
    const char* skip_passthrough_output_names[] = {"Y", "SkipSum"};
    const char* const* input_names = has_skip ? skip_input_names : simplified_input_names;
    const char* const* output_names = has_skip_passthrough ? skip_passthrough_output_names : main_output_names;
    const size_t input_count = has_skip ? 2u : 1u;
    const size_t output_count = has_skip_passthrough ? 2u : 1u;
    std::array<Ort::Value, 2> input_tensors = {
        Ort::Value::CreateTensor<Ort::Float16_t>(memory_info,
                                                 activation.data(),
                                                 activation.size(),
                                                 input_shape.data(),
                                                 input_shape.size()),
        Ort::Value::CreateTensor<Ort::Float16_t>(memory_info,
                                                 skip_activation.data(),
                                                 skip_activation.size(),
                                                 input_shape.data(),
                                                 input_shape.size())};
    Ort::RunOptions run_options = CreateBenchmarkRunOptions();

    if (!IsDecodeBenchmarkPerfMode()) {
      ValidateMlpDecodeOutputs(SerializeMatMulNBitsMlpModel(config, MlpDecodeBenchmarkVariant::kUnfused, norm_kind),
                               SerializeMatMulNBitsMlpModel(config, variant, norm_kind),
                               selected_context.provider_options,
                               input_names,
                               input_tensors.data(),
                               input_count,
                               output_names,
                               output_count);
    }

    for (int i = 0; i < kDecodeWarmupRuns; ++i) {
      auto warmup_outputs = session.Run(run_options, input_names, input_tensors.data(), input_count, output_names, output_count);
      benchmark::DoNotOptimize(warmup_outputs);
    }

    double total_kernel_seconds = 0.0;
    for (auto _ : state) {
      const auto kernel_start = std::chrono::steady_clock::now();
      auto outputs = session.Run(run_options, input_names, input_tensors.data(), input_count, output_names, output_count);
      const auto kernel_end = std::chrono::steady_clock::now();
      total_kernel_seconds += std::chrono::duration<double>(kernel_end - kernel_start).count();
      benchmark::DoNotOptimize(outputs);
    }

    const double total_flops = 4.0 * static_cast<double>(config.n) * static_cast<double>(config.k);
    const double achieved_bandwidth_bytes_per_second =
        total_kernel_seconds > 0.0
            ? traffic.total_bytes * static_cast<double>(state.iterations()) / total_kernel_seconds
            : 0.0;

    state.SetLabel(GetMlpDecodeBenchmarkLabel(variant, norm_kind));
    state.counters["TFLOPS"] = benchmark::Counter(
        total_flops,
        benchmark::Counter::kIsIterationInvariantRate);
    state.counters["ApproxMemBW_GBps"] = benchmark::Counter(achieved_bandwidth_bytes_per_second / 1.0e9);
    state.counters["ApproxTraffic_MB"] = benchmark::Counter(traffic.total_bytes / 1.0e6);
    state.counters["Input_MB"] = benchmark::Counter(traffic.input_bytes / 1.0e6);
    state.counters["PackedW_MB"] = benchmark::Counter(traffic.packed_weight_bytes / 1.0e6);
    state.counters["Scales_MB"] = benchmark::Counter(traffic.scale_bytes / 1.0e6);
    state.counters["Intermediate_MB"] = benchmark::Counter(traffic.intermediate_bytes / 1.0e6);
    state.counters["Output_MB"] = benchmark::Counter(traffic.output_bytes / 1.0e6);
    state.counters["GraphReplay"] = benchmark::Counter(IsGraphCaptureBenchmarkEnabled() ? 1.0 : 0.0);
  } catch (const std::exception& ex) {
    state.SkipWithError(ex.what());
  }
}

static void BM_WebGpuMatMulNBitsMlpSimplifiedDecodeUnfused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsMlpDecode(state, MlpDecodeBenchmarkVariant::kUnfused, MlpNormKind::kSimplified);
}

static void BM_WebGpuMatMulNBitsMlpSimplifiedDecodeFused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsMlpDecode(state, MlpDecodeBenchmarkVariant::kFused, MlpNormKind::kSimplified);
}

static void BM_WebGpuMatMulNBitsQkvSimplifiedNormDecodeUnfused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsQkvDecode(state, QkvDecodeBenchmarkVariant::kUnfused, QkvNormKind::kSimplified);
}

static void BM_WebGpuMatMulNBitsQkvSimplifiedNormDecodeFused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsQkvDecode(state, QkvDecodeBenchmarkVariant::kFused, QkvNormKind::kSimplified);
}

static void BM_WebGpuMatMulNBitsQkvSkipDecodeUnfused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsQkvDecode(state, QkvDecodeBenchmarkVariant::kUnfused, QkvNormKind::kSkipSimplified);
}

static void BM_WebGpuMatMulNBitsQkvSkipDecodeFused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsQkvDecode(state, QkvDecodeBenchmarkVariant::kFused, QkvNormKind::kSkipSimplified);
}

static void BM_WebGpuMatMulNBitsQkvSkipPassthroughDecodeUnfused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsQkvDecode(state, QkvDecodeBenchmarkVariant::kUnfused, QkvNormKind::kSkipSimplifiedPassthrough);
}

static void BM_WebGpuMatMulNBitsQkvSkipPassthroughDecodeFused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsQkvDecode(state, QkvDecodeBenchmarkVariant::kFused, QkvNormKind::kSkipSimplifiedPassthrough);
}

static void BM_WebGpuMatMulNBitsMlpSkipDecodeUnfused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsMlpDecode(state, MlpDecodeBenchmarkVariant::kUnfused, MlpNormKind::kSkipSimplified);
}

static void BM_WebGpuMatMulNBitsMlpSkipDecodeFused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsMlpDecode(state, MlpDecodeBenchmarkVariant::kFused, MlpNormKind::kSkipSimplified);
}

static void BM_WebGpuMatMulNBitsMlpSkipPassthroughDecodeUnfused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsMlpDecode(state, MlpDecodeBenchmarkVariant::kUnfused, MlpNormKind::kSkipSimplifiedPassthrough);
}

static void BM_WebGpuMatMulNBitsMlpSkipPassthroughDecodeFused(benchmark::State& state) {
  BenchmarkWebGpuMatMulNBitsMlpDecode(state, MlpDecodeBenchmarkVariant::kFused, MlpNormKind::kSkipSimplifiedPassthrough);
}

void ApplyWebGpuMatMulNBitsMlpDecodeArgs(benchmark::internal::Benchmark* benchmark) {
  for (const auto& config : GetMlpDecodeBenchConfigs()) {
    benchmark->Args({config.n, config.k, config.bits, config.block_size, config.accuracy_level});
  }
}

void ApplyWebGpuMatMulNBitsQkvDecodeArgs(benchmark::internal::Benchmark* benchmark) {
  for (const auto& config : GetQkvDecodeBenchConfigs()) {
    benchmark->Args({config.q_n, config.kv_n, config.k, config.bits, config.block_size, config.accuracy_level});
  }
}

// Qkv benchmarks
BENCHMARK(BM_WebGpuMatMulNBitsQkvSimplifiedNormDecodeUnfused)
    ->Apply(ApplyWebGpuMatMulNBitsQkvDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsQkvSimplifiedNormDecodeFused)
    ->Apply(ApplyWebGpuMatMulNBitsQkvDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsQkvSkipDecodeUnfused)
    ->Apply(ApplyWebGpuMatMulNBitsQkvDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsQkvSkipDecodeFused)
    ->Apply(ApplyWebGpuMatMulNBitsQkvDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsQkvSkipPassthroughDecodeUnfused)
    ->Apply(ApplyWebGpuMatMulNBitsQkvDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsQkvSkipPassthroughDecodeFused)
    ->Apply(ApplyWebGpuMatMulNBitsQkvDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

// Mlp benchmarks
BENCHMARK(BM_WebGpuMatMulNBitsMlpSimplifiedDecodeUnfused)
    ->Apply(ApplyWebGpuMatMulNBitsMlpDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsMlpSimplifiedDecodeFused)
    ->Apply(ApplyWebGpuMatMulNBitsMlpDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsMlpSkipDecodeUnfused)
    ->Apply(ApplyWebGpuMatMulNBitsMlpDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsMlpSkipDecodeFused)
    ->Apply(ApplyWebGpuMatMulNBitsMlpDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsMlpSkipPassthroughDecodeUnfused)
    ->Apply(ApplyWebGpuMatMulNBitsMlpDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_WebGpuMatMulNBitsMlpSkipPassthroughDecodeFused)
    ->Apply(ApplyWebGpuMatMulNBitsMlpDecodeArgs)
    ->ReportAggregatesOnly()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

}  // namespace
