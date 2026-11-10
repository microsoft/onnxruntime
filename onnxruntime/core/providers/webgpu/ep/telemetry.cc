// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/ep/telemetry.h"

#include <atomic>

#if defined(_WIN32)

#include <string>

// need space after Windows.h to prevent clang-format re-ordering breaking the build.
// TraceLoggingProvider.h must follow Windows.h
#include <Windows.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)  // Warning C26440 from TRACELOGGING_DEFINE_PROVIDER
#endif

#include <TraceLoggingProvider.h>
#include <winmeta.h>

// The private configuration defines TraceLoggingOptionMicrosoftTelemetry() to join the Microsoft telemetry provider
// group so that events are uploaded. It is supplied only by official builds; see onnxruntime_providers_webgpu.cmake.
#if defined(ORT_WEBGPU_EP_TRACELOGGING_CONFIG_HEADER)
#include ORT_WEBGPU_EP_TRACELOGGING_CONFIG_HEADER
#endif

// Without the private configuration, the provider joins no provider group and events are visible to local ETW only.
#ifndef TraceLoggingOptionMicrosoftTelemetry
#define TraceLoggingOptionMicrosoftTelemetry() \
  TraceLoggingOptionGroup(0000000000, 00000, 00000, 0000, 0000, 0000, 0000, 0000, 000, 0000, 0000)
#endif
#ifndef TelemetryPrivacyDataTag
#define TelemetryPrivacyDataTag(tag) TraceLoggingUInt64((tag), "PartA_PrivTags")
#endif
#ifndef PDT_DeviceConnectivityAndConfiguration
#define PDT_DeviceConnectivityAndConfiguration 0x0000000000000800u
#endif
#ifndef PDT_ProductAndServicePerformance
#define PDT_ProductAndServicePerformance 0x0000000001000000u
#endif
#ifndef PDT_ProductAndServiceUsage
#define PDT_ProductAndServiceUsage 0x0000000002000000u
#endif
#ifndef MICROSOFT_KEYWORD_MEASURES
#define MICROSOFT_KEYWORD_MEASURES 0x0000400000000000
#endif

// Older TraceLoggingProvider.h versions are incompatible with /utf-8.
#ifdef _TlgPragmaUtf8Begin
#undef _TlgPragmaUtf8Begin
#define _TlgPragmaUtf8Begin
#endif
#ifdef _TlgPragmaUtf8End
#undef _TlgPragmaUtf8End
#define _TlgPragmaUtf8End
#endif
#ifdef _tlgPragmaUtf8Begin
#undef _tlgPragmaUtf8Begin
#define _tlgPragmaUtf8Begin
#endif
#ifdef _tlgPragmaUtf8End
#undef _tlgPragmaUtf8End
#define _tlgPragmaUtf8End
#endif

#include <optional>

#include "ep/api.h"

#include "core/providers/webgpu/ep/telemetry_privacy.h"
#include "core/providers/webgpu/webgpu_external_header.h"

#if defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
// The statically linked variant is part of the ONNX Runtime binary, so this is not a cross-repository dependency.
#include "core/platform/env.h"
#include "core/platform/telemetry.h"
#endif

namespace onnxruntime {
namespace webgpu {
namespace ep {
namespace telemetry {

namespace {

// Separate DLLs cannot share a TraceLogging provider handle, so the plugin defines its own handle for the ONNX
// Runtime provider. This keeps WebGPU events in the same Asimov provider as the host's ONNX Runtime events.
TRACELOGGING_DEFINE_PROVIDER(webgpu_ep_telemetry_provider_handle, "Microsoft.ML.ONNXRuntime",
                             // {3a26b1ff-7484-7484-7484-15261f42614d}
                             (0x3a26b1ff, 0x7484, 0x7484, 0x74, 0x84, 0x15, 0x26, 0x1f, 0x42, 0x61, 0x4d),
                             TraceLoggingOptionMicrosoftTelemetry());

#ifdef _MSC_VER
#pragma warning(pop)
#endif

constexpr uint8_t kSchemaVersion = 0;

// Returns "" when the variable is not set, or nullopt when it cannot be read or is unexpectedly large.
std::optional<std::string> ReadEnvironmentVariable(const wchar_t* name) {
  constexpr DWORD kBufferSize = 256;
  wchar_t buffer[kBufferSize];
  ::SetLastError(ERROR_SUCCESS);
  const DWORD length = ::GetEnvironmentVariableW(name, buffer, kBufferSize);
  if (length == 0) {
    const DWORD error = ::GetLastError();
    if (error == ERROR_SUCCESS || error == ERROR_ENVVAR_NOT_FOUND) {
      return std::string{};
    }
    return std::nullopt;
  }
  if (length >= kBufferSize) {
    return std::nullopt;
  }
  // Only ASCII tokens are meaningful; any other character cannot match and is replaced.
  std::string value;
  value.reserve(length);
  for (DWORD i = 0; i < length; ++i) {
    value += buffer[i] < 0x80 ? static_cast<char>(buffer[i]) : '?';
  }
  return value;
}

// The host's OrtApi::DisableTelemetryEvents() state is not visible to plugins, so honor the environment opt-out.
// ORT unit-test processes never register, matching the host's telemetry providers. Unreadable values disable.
bool IsDisabledByEnvironment() {
  const auto running_unit_tests = ReadEnvironmentVariable(L"ORT_RUNNING_UNIT_TESTS");
  if (!running_unit_tests || IsNonFalseEnvironmentValue(*running_unit_tests)) {
    return true;
  }
  const auto disable_telemetry = ReadEnvironmentVariable(L"ORT_DISABLE_TELEMETRY");
  return !disable_telemetry || IsTrueEnvironmentValue(*disable_telemetry);
}

class ProviderRegistration {
 public:
  static const ProviderRegistration& Instance() {
    static const ProviderRegistration registration;
    return registration;
  }

  bool IsRegistered() const { return registered_; }

  ProviderRegistration(const ProviderRegistration&) = delete;
  ProviderRegistration& operator=(const ProviderRegistration&) = delete;

 private:
  ProviderRegistration() {
    if (IsDisabledByEnvironment()) {
      return;
    }
    registered_ = SUCCEEDED(TraceLoggingRegister(webgpu_ep_telemetry_provider_handle));
  }

  ~ProviderRegistration() {
    if (registered_) {
      TraceLoggingUnregister(webgpu_ep_telemetry_provider_handle);
    }
  }

  bool registered_ = false;
};

bool ShouldLog() noexcept {
  try {
#if defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
    // Linked into the ONNX Runtime binary, the plugin can and must honor the host's telemetry state, including
    // OrtApi::DisableTelemetryEvents(). A separately loaded plugin cannot observe that state.
    if (!onnxruntime::Env::Default().GetTelemetryProvider().IsEnabled()) {
      return false;
    }
#endif
    return ProviderRegistration::Instance().IsRegistered() &&
           TraceLoggingProviderEnabled(webgpu_ep_telemetry_provider_handle, 0, MICROSOFT_KEYWORD_MEASURES);
  } catch (...) {
    return false;
  }
}

const char* HostOrtVersion() {
  static const std::string version = []() -> std::string {
    try {
      return "1." + std::to_string(onnxruntime::ep::CurrentOrtApiVersion()) + "." +
             std::to_string(onnxruntime::ep::CurrentOrtPatchVersion());
    } catch (...) {
      return {};
    }
  }();
  return version.c_str();
}

const char* BackendTypeName(uint32_t backend_type) {
  switch (static_cast<wgpu::BackendType>(backend_type)) {
    case wgpu::BackendType::Undefined:
      return "Undefined";
    case wgpu::BackendType::Null:
      return "Null";
    case wgpu::BackendType::WebGPU:
      return "WebGPU";
    case wgpu::BackendType::D3D11:
      return "D3D11";
    case wgpu::BackendType::D3D12:
      return "D3D12";
    case wgpu::BackendType::Metal:
      return "Metal";
    case wgpu::BackendType::Vulkan:
      return "Vulkan";
    case wgpu::BackendType::OpenGL:
      return "OpenGL";
    case wgpu::BackendType::OpenGLES:
      return "OpenGLES";
    default:
      return "Other";
  }
}

const char* AdapterTypeName(uint32_t adapter_type) {
  switch (static_cast<wgpu::AdapterType>(adapter_type)) {
    case wgpu::AdapterType::DiscreteGPU:
      return "DiscreteGPU";
    case wgpu::AdapterType::IntegratedGPU:
      return "IntegratedGPU";
    case wgpu::AdapterType::CPU:
      return "CPU";
    case wgpu::AdapterType::Unknown:
      return "Unknown";
    default:
      return "Other";
  }
}

const char* DeviceErrorKindName(DeviceErrorKind kind) {
  return kind == DeviceErrorKind::kDeviceLost ? "deviceLost" : "uncapturedError";
}

}  // namespace

bool IsEnabled() noexcept {
  return ShouldLog();
}

uint32_t NextEpInstanceId() noexcept {
  static std::atomic<uint32_t> next_id{0};
  uint32_t id = ++next_id;
  // Zero is reserved for "unknown".
  return id != 0 ? id : ++next_id;
}

void LogFactoryCreated(const FactoryCreatedInfo& info) noexcept {
  try {
    if (!ShouldLog()) {
      return;
    }
#if defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
    constexpr bool statically_linked = true;
#else
    constexpr bool statically_linked = false;
#endif
    TraceLoggingWrite(webgpu_ep_telemetry_provider_handle,
                      "WebGpuEpFactoryCreated",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      TraceLoggingUInt8(kSchemaVersion, "schemaVersion"),
                      TraceLoggingString(ORT_PLUGIN_EP_VERSION, "epVersion"),
                      TraceLoggingString(HostOrtVersion(), "hostOrtVersion"),
                      TraceLoggingString(info.serialized_execution_mode ? "serialized" : "concurrent",
                                         "executionMode"),
                      TraceLoggingBoolean(info.serialized_execution_mode_forced, "serializedModeForced"),
                      TraceLoggingBoolean(info.allow_software_adapter, "allowSoftwareAdapter"),
                      TraceLoggingBoolean(info.allow_virtual_devices, "allowVirtualDevices"),
                      TraceLoggingBoolean(statically_linked, "staticallyLinked"));
  } catch (...) {
  }
}

void LogEpCreation(const EpCreationInfo& info) noexcept {
  try {
    if (!ShouldLog()) {
      return;
    }
    // TraceLoggingWrite evaluates arguments across several statements, so temporaries cannot own these strings.
    const std::string error_message = ScrubPaths(info.error_message);
    const std::string gpu_vendor = BoundedUtf8(info.gpu_vendor);
    const std::string gpu_architecture = BoundedUtf8(info.gpu_architecture);
    const std::string gpu_device = BoundedUtf8(info.gpu_device);
    const std::string gpu_description = BoundedUtf8(info.gpu_description);
    TraceLoggingWrite(webgpu_ep_telemetry_provider_handle,
                      "WebGpuEpCreation",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage | PDT_ProductAndServicePerformance |
                                              PDT_DeviceConnectivityAndConfiguration),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      TraceLoggingUInt8(kSchemaVersion, "schemaVersion"),
                      TraceLoggingString(ORT_PLUGIN_EP_VERSION, "epVersion"),
                      TraceLoggingString(HostOrtVersion(), "hostOrtVersion"),
                      TraceLoggingUInt32(info.ep_instance_id, "epInstanceId"),
                      TraceLoggingBoolean(info.success, "success"),
                      TraceLoggingUInt32(info.error_code, "errorCode"),
                      TraceLoggingString(error_message.c_str(), "errorMessage"),
                      TraceLoggingInt64(info.duration_us, "durationUs"),
                      TraceLoggingBoolean(info.compile_only, "compileOnly"),
                      TraceLoggingBoolean(info.virtual_device, "virtualDevice"),
                      TraceLoggingUInt32(info.hw_device_type, "hardwareDeviceType"),
                      TraceLoggingUInt32(info.hw_vendor_id, "hardwareVendorId"),
                      TraceLoggingUInt32(info.hw_device_id, "hardwareDeviceId"),
                      TraceLoggingBoolean(info.has_device, "hasDevice"),
                      TraceLoggingString(info.has_device ? BackendTypeName(info.backend_type) : "", "backendType"),
                      TraceLoggingString(info.has_device ? AdapterTypeName(info.adapter_type) : "", "adapterType"),
                      TraceLoggingUInt32(info.gpu_vendor_id, "gpuVendorId"),
                      TraceLoggingUInt32(info.gpu_device_id, "gpuDeviceId"),
                      TraceLoggingString(gpu_vendor.c_str(), "gpuVendor"),
                      TraceLoggingString(gpu_architecture.c_str(), "gpuArchitecture"),
                      TraceLoggingString(gpu_device.c_str(), "gpuDevice"),
                      TraceLoggingString(gpu_description.c_str(), "gpuDescription"),
                      TraceLoggingUInt32(info.subgroup_min_size, "subgroupMinSize"),
                      TraceLoggingUInt32(info.subgroup_max_size, "subgroupMaxSize"),
                      TraceLoggingBoolean(info.feature_shader_f16, "featureShaderF16"),
                      TraceLoggingBoolean(info.feature_subgroups, "featureSubgroups"),
                      TraceLoggingBoolean(info.feature_subgroup_matrix, "featureSubgroupMatrix"),
                      TraceLoggingBoolean(info.feature_timestamp_query, "featureTimestampQuery"),
                      TraceLoggingUInt64(info.max_buffer_size, "maxBufferSize"),
                      TraceLoggingUInt64(info.max_storage_buffer_binding_size, "maxStorageBufferBindingSize"),
                      TraceLoggingUInt32(info.max_compute_workgroup_storage_size, "maxComputeWorkgroupStorageSize"),
                      TraceLoggingUInt32(info.max_compute_invocations_per_workgroup,
                                         "maxComputeInvocationsPerWorkgroup"),
                      TraceLoggingString(info.nhwc_layout ? "NHWC" : "NCHW", "dataLayout"),
                      TraceLoggingBoolean(info.graph_capture_enabled, "graphCaptureEnabled"),
                      TraceLoggingUInt32(info.validation_mode, "validationMode"),
                      TraceLoggingUInt32(info.kv_cache_quantization_bits, "kvCacheQuantizationBits"),
                      TraceLoggingBoolean(info.matmul_fp32_accumulation, "matmulFp32Accumulation"),
                      TraceLoggingBoolean(info.forced_matmul_algorithm, "forcedMatMulAlgorithm"),
                      TraceLoggingUInt32(info.force_cpu_node_count, "forceCpuNodeCount"));
  } catch (...) {
  }
}

void LogGraphPartition(const GraphPartitionInfo& info) noexcept {
  try {
    if (!ShouldLog()) {
      return;
    }
    const std::string no_kernel_op_types = BoundedUtf8(info.no_kernel_op_types);
    const std::string unsupported_config_op_types = BoundedUtf8(info.unsupported_config_op_types);
    TraceLoggingWrite(webgpu_ep_telemetry_provider_handle,
                      "WebGpuGraphPartition",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage | PDT_ProductAndServicePerformance),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      TraceLoggingUInt8(kSchemaVersion, "schemaVersion"),
                      TraceLoggingString(ORT_PLUGIN_EP_VERSION, "epVersion"),
                      TraceLoggingString(HostOrtVersion(), "hostOrtVersion"),
                      TraceLoggingUInt32(info.ep_instance_id, "epInstanceId"),
                      TraceLoggingUInt32(info.pass_index, "passIndex"),
                      TraceLoggingUInt32(info.total_nodes, "totalNodes"),
                      TraceLoggingUInt32(info.claimed_nodes, "claimedNodes"),
                      TraceLoggingUInt32(info.preassigned_nodes, "preassignedNodes"),
                      TraceLoggingUInt32(info.other_ep_nodes, "otherEpNodes"),
                      TraceLoggingUInt32(info.no_kernel_nodes, "noKernelNodes"),
                      TraceLoggingUInt32(info.forced_cpu_nodes, "forcedCpuNodes"),
                      TraceLoggingUInt32(info.unsupported_config_nodes, "unsupportedConfigNodes"),
                      TraceLoggingUInt32(info.cpu_preferred_nodes, "cpuPreferredNodes"),
                      TraceLoggingString(no_kernel_op_types.c_str(), "noKernelOpTypes"),
                      TraceLoggingUInt32(info.no_kernel_op_type_count, "noKernelOpTypeCount"),
                      TraceLoggingString(unsupported_config_op_types.c_str(), "unsupportedConfigOpTypes"));
  } catch (...) {
  }
}

void LogDeviceError(const DeviceErrorInfo& info) noexcept {
  try {
    if (!ShouldLog()) {
      return;
    }
    const std::string message = ScrubPaths(info.message);
    TraceLoggingWrite(webgpu_ep_telemetry_provider_handle,
                      "WebGpuDeviceError",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
                      TraceLoggingUInt8(kSchemaVersion, "schemaVersion"),
                      TraceLoggingString(ORT_PLUGIN_EP_VERSION, "epVersion"),
                      TraceLoggingString(HostOrtVersion(), "hostOrtVersion"),
                      TraceLoggingString(DeviceErrorKindName(info.kind), "errorKind"),
                      TraceLoggingUInt32(info.reason_code, "reasonCode"),
                      TraceLoggingString(message.c_str(), "errorMessage"),
                      TraceLoggingString(info.backend_type != 0 ? BackendTypeName(info.backend_type) : "",
                                         "backendType"),
                      TraceLoggingUInt32(info.gpu_vendor_id, "gpuVendorId"),
                      TraceLoggingUInt32(info.gpu_device_id, "gpuDeviceId"));
  } catch (...) {
  }
}

void LogRunSummary(const RunSummaryInfo& info) noexcept {
  try {
    if (!ShouldLog()) {
      return;
    }
    TraceLoggingWrite(webgpu_ep_telemetry_provider_handle,
                      "WebGpuRunSummary",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage | PDT_ProductAndServicePerformance),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      TraceLoggingUInt8(kSchemaVersion, "schemaVersion"),
                      TraceLoggingString(ORT_PLUGIN_EP_VERSION, "epVersion"),
                      TraceLoggingString(HostOrtVersion(), "hostOrtVersion"),
                      TraceLoggingUInt32(info.ep_instance_id, "epInstanceId"),
                      TraceLoggingBoolean(info.is_final, "isFinal"),
                      TraceLoggingUInt32(info.runs, "runs"),
                      TraceLoggingUInt32(info.run_start_failures, "runStartFailures"),
                      TraceLoggingUInt32(info.run_end_failures, "runEndFailures"),
                      TraceLoggingInt64(info.first_run_duration_us, "firstRunDurationUs"),
                      TraceLoggingInt64(info.total_run_duration_us, "totalRunDurationUs"),
                      TraceLoggingInt64(info.max_run_duration_us, "maxRunDurationUs"),
                      TraceLoggingUInt32(info.graph_replays, "graphReplays"),
                      TraceLoggingUInt32(info.graph_replay_failures, "graphReplayFailures"),
                      TraceLoggingUInt64(info.program_cache_size, "programCacheSize"));
  } catch (...) {
  }
}

}  // namespace telemetry
}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime

#else  // !defined(_WIN32)

namespace onnxruntime {
namespace webgpu {
namespace ep {
namespace telemetry {

bool IsEnabled() noexcept { return false; }

uint32_t NextEpInstanceId() noexcept {
  static std::atomic<uint32_t> next_id{0};
  uint32_t id = ++next_id;
  return id != 0 ? id : ++next_id;
}

void LogFactoryCreated(const FactoryCreatedInfo& /*info*/) noexcept {}
void LogEpCreation(const EpCreationInfo& /*info*/) noexcept {}
void LogGraphPartition(const GraphPartitionInfo& /*info*/) noexcept {}
void LogDeviceError(const DeviceErrorInfo& /*info*/) noexcept {}
void LogRunSummary(const RunSummaryInfo& /*info*/) noexcept {}

}  // namespace telemetry
}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // defined(_WIN32)
