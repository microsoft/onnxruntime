// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//
// WinMLTelemetryHelper provides a centralized location for managing all telemetry
// usage in the WinML COM runtime.  This aims to abstract all interaction with the
// TraceLogging APIs.
//
// A global instance of the helper is declared in precomp.h and defined in dll.cpp.
//

// TraceLogging includes
#include <winmeta.h>
#include <TraceLoggingProvider.h>

// Forward references
class WinMLRuntime;
typedef struct WinMLModelDescription {
  LPWSTR Author;
  LPWSTR Name;
  LPWSTR Domain;
  LPWSTR Description;
  SIZE_T Version;
} WinMLModelDescription;

template <typename T>
class Profiler;

// Schema versions.
#define WINML_TLM_PROCESS_INFO_SCHEMA_VERSION 0
#define WINML_TLM_CONTEXT_CREATION_VERSION 0
#define WINML_TLM_MODEL_CREATION_VERSION 0
#define WINML_TLM_RUNTIME_ERROR_VERSION 0
#define WINML_TLM_RUNTIME_PERF_VERSION 0
#define WINML_TLM_NATIVE_API_INTRAOP_THREADS_VERSION 0
#define WINML_TLM_NATIVE_API_INTRAOP_THREAD_SPINNING_VERSION 0
#define WINML_TLM_NAMED_DIMENSION_OVERRIDE_VERSION 0
#define WINML_TLM_EXPERIMENTAL_API_VERSION 0

#define WinMLTraceLoggingWrite(hProvider, EventName, ...)                \
  TraceLoggingWrite(hProvider,                                           \
                    EventName,                                           \
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"), \
                    __VA_ARGS__)
//
// WinMLRuntime Telemetry Support
//
// {BCAD6AEE-C08D-4F66-828C-4C43461A033D}
#define WINML_PROVIDER_DESC "Microsoft.Windows.AI.MachineLearning"
#define WINML_PROVIDER_GUID (0xbcad6aee, 0xc08d, 0x4f66, 0x82, 0x8c, 0x4c, 0x43, 0x46, 0x1a, 0x3, 0x3d)
#define WINML_PROVIDER_KEYWORD_DEFAULT 0x1
#define WINML_PROVIDER_KEYWORD_LOTUS_PROFILING 0x2
#define WINML_PROVIDER_KEYWORD_START_STOP 0x4
struct MLOperatorKernelDescription;
struct MLOperatorSchemaDescription;

class WinMLTelemetryHelper {
 public:
  TraceLoggingHProvider provider_ = nullptr;
  // Flag indicating the success of registering our telemetry provider.
  bool telemetry_enabled_ = false;

  WinMLTelemetryHelper();
  ~WinMLTelemetryHelper();

  //
  // Register telemetry provider and check success.  Will only succeed if
  // client has opted in to sending MS telemetry.
  //
  virtual HRESULT Register() {
    HRESULT hr = TraceLoggingRegister(provider_);
    if (SUCCEEDED(hr)) {
      telemetry_enabled_ = true;
    }
    return hr;
  }

  //
  // Un-Register telemetry provider to ignore events from a TraceLogging provider.
  //
  void UnRegister() {
    TraceLoggingUnregister(provider_);
  }

  void LogApiUsage(const char* name);
  void LogWinMLShutDown();
  void LogRuntimeError(HRESULT hr, std::string message, PCSTR file, PCSTR function, int line);
  void LogRuntimeError(HRESULT hr, PCSTR message, PCSTR file, PCSTR function, int line);
  void LogRegisterOperatorKernel(
      const char* name,
      const char* domain,
      int execution_type);
  void RegisterOperatorSetSchema(
      const char* name,
      uint32_t input_count,
      uint32_t output_count,
      uint32_t type_constraint_count,
      uint32_t attribute_count,
      uint32_t default_attribute_count);
  void SetIntraOpNumThreadsOverride(
      uint32_t num_threads);
  void SetIntraOpThreadSpinning(
      bool allow_spinning);
  void SetNamedDimensionOverride(
      winrt::hstring name,
      uint32_t value);
  void EndRuntimeSession() { ++runtime_session_id_; };
  bool IsMeasureSampled();
  int GetRuntimeSessionId() { return runtime_session_id_; }

 private:
  void RestartTimer() {
    timer_start_ = GetTickCount64();
    timer_started_ = true;
  }

 private:
  int runtime_session_id_;
  unsigned int log_counter_ = 0;

  bool timer_started_ = false;
  ULONGLONG timer_start_ = 0;
};
