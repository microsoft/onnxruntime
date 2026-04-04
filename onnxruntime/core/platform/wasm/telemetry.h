// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/telemetry.h"

#include <atomic>

namespace onnxruntime {

/**
 * @brief WebAssembly telemetry implementation.
 *
 * Bridges C++ telemetry events to JavaScript via Emscripten EM_JS.
 * The JS side receives structured events and forwards them to the 1DS SDK
 * and/or observer callbacks configured via ort.env.telemetry.
 */
class WasmTelemetry : public Telemetry {
 public:
  WasmTelemetry() = default;
  ~WasmTelemetry() override = default;

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
                      const std::unordered_map<int64_t, long long>& duration_per_batch_size) const override;

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

  void LogModelLoadStart(uint32_t session_id) const override;
  void LogModelLoadEnd(uint32_t session_id, const common::Status& status) const override;

  void LogSessionCreationEnd(uint32_t session_id, const common::Status& status) const override;

  void LogRegisterEpLibraryWithLibPath(const std::string& registration_name,
                                       const std::string& lib_path) const override;
  void LogRegisterEpLibraryStart(const std::string& registration_name) const override;
  void LogRegisterEpLibraryEnd(const std::string& registration_name,
                               const common::Status& status) const override;

 private:
  mutable std::atomic<bool> enabled_{true};
  mutable std::atomic<uint32_t> projection_{0};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WasmTelemetry);
};

}  // namespace onnxruntime
