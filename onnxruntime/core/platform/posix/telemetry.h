// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef _WIN32  // Only for non-Windows platforms

#include "core/platform/telemetry.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations of 1DS SDK types (must be at global scope)
namespace Microsoft::Applications::Events {
class ILogger;
class ISemanticContext;
class EventProperties;
}  // namespace Microsoft::Applications::Events

namespace onnxruntime {

/**
* @brief Telemetry implementation for non-Windows platforms.
* 
* This class provides telemetry logging capabilities for macOS, Linux, Android, and iOS
* using the cpp_client_telemetry library (1DS SDK). It implements the same interface
* as WindowsTelemetry to provide consistent telemetry across all platforms.
* 
* Configuration:
* - Telemetry is opt-in via build flags
*/
class PosixTelemetry : public Telemetry {
 public:
  PosixTelemetry();
  ~PosixTelemetry() override;

  void EnableTelemetryEvents() const override;
  void DisableTelemetryEvents() const override;
  void SetLanguageProjection(uint32_t projection) const override;

  bool IsEnabled() const override;
  unsigned char Level() const override;
  uint64_t Keyword() const override;

  void LogProcessInfo() const override;
  void LogSessionCreationStart(uint32_t session_id) const override;
  void LogEvaluationStop(uint32_t session_id) const override;
  void LogEvaluationStart(uint32_t session_id) const override;

  void LogSessionCreation(uint32_t session_id, int64_t ir_version,
                         const std::string& model_producer_name,
                         const std::string& model_producer_version,
                         const std::string& model_domain,
                         const std::unordered_map<std::string, int>& domain_to_version_map,
                         const std::string& model_file_name,
                         const std::string& model_graph_name,
                         const std::string& model_weight_type,
                         const std::string& model_graph_hash,
                         const std::string& model_weight_hash,
                         const std::unordered_map<std::string, std::string>& model_metadata,
                         const std::string& loadedFrom,
                         const std::vector<std::string>& execution_provider_ids,
                         bool use_fp16, bool captureState) const override;

  void LogCompileModelStart(uint32_t session_id,
                           const std::string& input_source,
                           const std::string& output_target,
                           uint32_t flags,
                           int graph_optimization_level,
                           bool embed_ep_context,
                           bool has_external_initializers_file,
                           const std::vector<std::string>& execution_provider_ids) const override;

  void LogCompileModelComplete(uint32_t session_id,
                               bool success,
                               uint32_t error_code,
                               uint32_t error_category,
                               const std::string& error_message) const override;

  void LogRuntimeError(uint32_t session_id, const common::Status& status,
                      const char* file, const char* function, uint32_t line) const override;

  void LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last,
                     int64_t total_run_duration_since_last,
                     std::unordered_map<int64_t, long long> duration_per_batch_size) const override;

  void LogExecutionProviderEvent(LUID* adapterLuid) const override;
  void LogDriverInfoEvent(const std::string_view device_class,
                         const std::wstring_view& driver_names,
                         const std::wstring_view& driver_versions) const override;

  void LogAutoEpSelection(uint32_t session_id, const std::string& selection_policy,
                         const std::vector<std::string>& requested_execution_provider_ids,
                         const std::vector<std::string>& available_execution_provider_ids) const override;

  void LogProviderOptions(const std::string& provider_id,
                         const std::string& provider_options_string,
                         bool captureState) const override;

 private:
  // Initialize telemetry SDK logger
  void Initialize();
  
  // Shutdown telemetry SDK logger
  void Shutdown();

  // Helper to get platform-specific information
  std::string GetPlatformInfo() const;
  std::string GetDeviceInfo() const;
  
  // Safe async event logging
  void LogEventAsync(::Microsoft::Applications::Events::EventProperties&& props) const;
  
  // Posix-specific: Log system resource metrics
  void LogPosixSystemMetrics(uint32_t session_id) const;

  // Mutex for thread-safe access
  mutable std::mutex mutex_;

  // Telemetry SDK logger instance (1DS)
  std::shared_ptr<::Microsoft::Applications::Events::ILogger> logger_;

  // State tracking
  mutable std::atomic<bool> enabled_{true};
  mutable std::atomic<uint32_t> projection_{0};
  mutable std::atomic<unsigned char> level_{0};
  mutable std::atomic<uint64_t> keyword_{0};

  // Process info tracking
  mutable std::atomic<bool> process_info_logged_{false};

  // Global registration count for singleton behavior
  static std::atomic<uint32_t> global_register_count_;
  static std::mutex global_mutex_;
  
  // Make EventBuilder a friend so it can access GetPlatformInfo/GetDeviceInfo
  friend class EventBuilder;
};

}  // namespace onnxruntime

#endif  // !_WIN32

