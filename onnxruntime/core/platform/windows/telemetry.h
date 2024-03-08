// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <vector>

#include "core/platform/telemetry.h"
#include <Windows.h>
#include <TraceLoggingProvider.h>
#include "core/platform/ort_mutex.h"
#include "core/platform/windows/TraceLoggingConfig.h"

namespace onnxruntime {

/**
 * derives and implments a Telemetry provider on Windows
 */
class WindowsTelemetry : public Telemetry {
 public:
  // these are allowed to be created, WindowsEnv will create one
  WindowsTelemetry();
  ~WindowsTelemetry();

  void EnableTelemetryEvents() const override;
  void DisableTelemetryEvents() const override;
  void SetLanguageProjection(uint32_t projection) const override;

  bool IsEnabled() const override;

  // Get the current logging level
  unsigned char Level() const override;

  // Get the current keyword
  UINT64 Keyword() const override;

  // Get the ETW registration status
  // static HRESULT Status();

  void LogProcessInfo() const override;

  void LogSessionCreationStart() const override;

  void LogEvaluationStop() const override;

  void LogEvaluationStart() const override;

  void LogSessionCreation(uint32_t session_id, int64_t ir_version, const std::string& model_producer_name,
                          const std::string& model_producer_version, const std::string& model_domain,
                          const std::unordered_map<std::string, int>& domain_to_version_map,
                          const std::string& model_graph_name,
                          const std::unordered_map<std::string, std::string>& model_metadata,
                          const std::string& loadedFrom, const std::vector<std::string>& execution_provider_ids,
                          bool use_fp16) const override;

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                       const char* function, uint32_t line) const override;

  void LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last) const override;

  void LogExecutionProviderEvent(LUID* adapterLuid) const override;

  using EtwInternalCallback = std::function<void(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level,
                                                 ULONGLONG MatchAnyKeyword, ULONGLONG MatchAllKeyword,
                                                 PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext)>;

  static void RegisterInternalCallback(const EtwInternalCallback& callback);

 private:
  static OrtMutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;
  static uint32_t projection_;

  static std::vector<EtwInternalCallback> callbacks_;
  static OrtMutex callbacks_mutex_;
  static OrtMutex provider_change_mutex_;
  static UCHAR level_;
  static ULONGLONG keyword_;

  static void InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                              ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData, PVOID CallbackContext);

  static void NTAPI ORT_TL_EtwEnableCallback(
      _In_ LPCGUID SourceId,
      _In_ ULONG IsEnabled,
      _In_ UCHAR Level,
      _In_ ULONGLONG MatchAnyKeyword,
      _In_ ULONGLONG MatchAllKeyword,
      _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData,
      _In_opt_ PVOID CallbackContext);
};

}  // namespace onnxruntime
