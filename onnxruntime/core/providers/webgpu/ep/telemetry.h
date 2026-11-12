// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>

namespace onnxruntime {
namespace webgpu {
namespace ep {
namespace telemetry {

// Telemetry for the WebGPU plugin EP.
//
// Events are written with Windows TraceLogging to the plugin's own provider ("Microsoft.ML.ONNXRuntime.WebGpuEp")
// using the MICROSOFT_KEYWORD_MEASURES keyword. Events are uploaded only when the build supplies the private
// TraceLogging configuration; otherwise they are visible to local ETW sessions only. Events are emitted only while the
// host permits non-essential telemetry (OrtEpApi::GetTelemetryEventsEnabled, ORT 1.33 and later), so nothing is
// emitted on older hosts. On other platforms all functions are no-ops.
//
// Every Log* function is noexcept and safe to call from Dawn callbacks.

struct FactoryCreatedInfo {
  bool serialized_execution_mode = false;
  bool serialized_execution_mode_forced = false;
  bool allow_software_adapter = false;
  bool allow_virtual_devices = false;
};

struct EpCreationInfo {
  uint32_t ep_instance_id = 0;
  bool success = false;
  uint32_t error_code = 0;
  std::string_view error_message;
  int64_t duration_us = 0;
  bool compile_only = false;
  bool virtual_device = false;

  // Hardware device selected by the host (OS enumeration).
  uint32_t hw_device_type = 0;
  uint32_t hw_vendor_id = 0;
  uint32_t hw_device_id = 0;

  // Dawn adapter, available only when the EP has a device.
  bool has_device = false;
  uint32_t backend_type = 0;
  uint32_t adapter_type = 0;
  uint32_t gpu_vendor_id = 0;
  uint32_t gpu_device_id = 0;
  std::string_view gpu_vendor;
  std::string_view gpu_architecture;
  std::string_view gpu_device;
  std::string_view gpu_description;
  uint32_t subgroup_min_size = 0;
  uint32_t subgroup_max_size = 0;
  bool feature_shader_f16 = false;
  bool feature_subgroups = false;
  bool feature_subgroup_matrix = false;
  bool feature_timestamp_query = false;
  uint64_t max_buffer_size = 0;
  uint64_t max_storage_buffer_binding_size = 0;
  uint32_t max_compute_workgroup_storage_size = 0;
  uint32_t max_compute_invocations_per_workgroup = 0;

  // Session configuration.
  bool nhwc_layout = false;
  bool graph_capture_enabled = false;
  uint32_t validation_mode = 0;
  uint32_t kv_cache_quantization_bits = 0;
  bool matmul_fp32_accumulation = false;
  bool forced_matmul_algorithm = false;
  uint32_t force_cpu_node_count = 0;
};

struct GraphPartitionInfo {
  uint32_t ep_instance_id = 0;
  uint32_t pass_index = 0;
  uint32_t total_nodes = 0;
  uint32_t claimed_nodes = 0;
  uint32_t preassigned_nodes = 0;
  uint32_t other_ep_nodes = 0;
  uint32_t no_kernel_nodes = 0;
  uint32_t forced_cpu_nodes = 0;
  uint32_t unsupported_config_nodes = 0;
  uint32_t cpu_preferred_nodes = 0;
  std::string_view no_kernel_op_types;
  uint32_t no_kernel_op_type_count = 0;
  std::string_view unsupported_config_op_types;
};

enum class DeviceErrorKind : uint32_t {
  kUncapturedError = 0,
  kDeviceLost = 1,
};

struct DeviceErrorInfo {
  DeviceErrorKind kind = DeviceErrorKind::kUncapturedError;
  uint32_t reason_code = 0;
  std::string_view message;
  uint32_t backend_type = 0;
  uint32_t gpu_vendor_id = 0;
  uint32_t gpu_device_id = 0;
};

struct RunSummaryInfo {
  uint32_t ep_instance_id = 0;
  bool is_final = false;
  uint32_t runs = 0;
  uint32_t run_start_failures = 0;
  uint32_t run_end_failures = 0;
  int64_t first_run_duration_us = 0;
  int64_t total_run_duration_us = 0;
  int64_t max_run_duration_us = 0;
  uint32_t graph_replays = 0;
  uint32_t graph_replay_failures = 0;
  uint64_t program_cache_size = 0;
};

// True when the host permits telemetry and events can be recorded by at least one listener. Use it to skip work that
// only feeds telemetry.
bool IsEnabled() noexcept;

// Returns a process-unique, nonzero identifier for correlating the events of one EP instance.
uint32_t NextEpInstanceId() noexcept;

void LogFactoryCreated(const FactoryCreatedInfo& info) noexcept;
void LogEpCreation(const EpCreationInfo& info) noexcept;
void LogGraphPartition(const GraphPartitionInfo& info) noexcept;
void LogDeviceError(const DeviceErrorInfo& info) noexcept;
void LogRunSummary(const RunSummaryInfo& info) noexcept;

}  // namespace telemetry
}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
