// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "OnnxruntimeEnvironment.h"
#include "OnnxruntimeErrors.h"
#include "core/platform/windows/TraceLoggingConfig.h"
#include <evntrace.h>
#include <windows.h>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.ApplicationModel.Core.h>

using namespace _winml;

static bool debug_output_ = false;

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

static std::string CurrentModulePath() {
  char path[MAX_PATH];
  FAIL_FAST_IF(0 == GetModuleFileNameA((HINSTANCE)&__ImageBase, path, _countof(path)));

  char absolute_path[MAX_PATH];
  char* name;
  FAIL_FAST_IF(0 == GetFullPathNameA(path, _countof(path), absolute_path, &name));

  auto idx = std::distance(absolute_path, name);
  auto out_path = std::string(absolute_path);
  out_path.resize(idx);

  return out_path;
}

static HRESULT GetOnnxruntimeLibrary(HMODULE& module) {
#if WINAPI_FAMILY == WINAPI_FAMILY_PC_APP
  // Store + Redist (note that this is never built into the inbox dll)
  auto out_module = LoadPackagedLibrary(L"onnxruntime.dll", 0);
#else
  auto onnxruntime_dll = CurrentModulePath() + "\\onnxruntime.dll"; 
  auto out_module = LoadLibraryExA(onnxruntime_dll.c_str(), nullptr, 0);
#endif

  if (out_module == nullptr) {
    return HRESULT_FROM_WIN32(GetLastError());
  }
  module = out_module;
  return S_OK;
}

const OrtApi* _winml::GetVersionedOrtApi() {
  HMODULE onnxruntime_dll;
  FAIL_FAST_IF_FAILED(GetOnnxruntimeLibrary(onnxruntime_dll));

  using OrtGetApiBaseSignature = decltype(OrtGetApiBase);
  auto ort_get_api_base_fn = reinterpret_cast<OrtGetApiBaseSignature*>(GetProcAddress(onnxruntime_dll, "OrtGetApiBase"));
  if (ort_get_api_base_fn == nullptr) {
    FAIL_FAST_HR(HRESULT_FROM_WIN32(GetLastError()));
  }

  const auto ort_api_base = ort_get_api_base_fn();

  static const uint32_t ort_version = 2;
  return ort_api_base->GetApi(ort_version);
}

static const WinmlAdapterApi* GetVersionedWinmlAdapterApi(const OrtApi* ort_api) {
  HMODULE onnxruntime_dll;
  FAIL_FAST_IF_FAILED(GetOnnxruntimeLibrary(onnxruntime_dll));

  using OrtGetWinMLAdapterSignature = decltype(OrtGetWinMLAdapter);
  auto ort_get_winml_adapter_fn = reinterpret_cast<OrtGetWinMLAdapterSignature*>(GetProcAddress(onnxruntime_dll, "OrtGetWinMLAdapter"));
  if (ort_get_winml_adapter_fn == nullptr) {
    FAIL_FAST_HR(HRESULT_FROM_WIN32(GetLastError()));
  }

  return ort_get_winml_adapter_fn(ort_api);
}

const WinmlAdapterApi* _winml::GetVersionedWinmlAdapterApi() {
  return GetVersionedWinmlAdapterApi(GetVersionedOrtApi());
}

static void __stdcall WinmlOrtLoggingCallback(void* param, OrtLoggingLevel severity, const char* category,
                                    const char* logger_id, const char* code_location, const char* message) noexcept {
  UNREFERENCED_PARAMETER(param);
  UNREFERENCED_PARAMETER(logger_id);
  // ORT Fatal and Error Messages are logged as Telemetry, rest are non-telemetry.
  switch (severity) {
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL:  //Telemetry
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_CRITICAL),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location),
          TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR:  //Telemetry
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location),
          TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING:
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_WARNING),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO:
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_INFO),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location));
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE:
      __fallthrough;  //Default is Verbose too.
    default:
      TraceLoggingWrite(
          winml_trace_logging_provider,
          "WinMLLogSink",
          TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
          TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
          TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
          TraceLoggingString(category),
          TraceLoggingUInt32((UINT32)severity),
          TraceLoggingString(message),
          TraceLoggingString(code_location));
  }

  if (debug_output_) {
    OutputDebugStringA((std::string(message) + "\r\n").c_str());
  }
}

static void __stdcall WinmlOrtProfileEventCallback(const OrtProfilerEventRecord* profiler_record) noexcept {
  if (profiler_record->category_ == OrtProfilerEventCategory::NODE_EVENT) {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "OnnxRuntimeProfiling",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
        TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
        TraceLoggingString(profiler_record->category_name_, "Category"),
        TraceLoggingInt64(profiler_record->duration_, "Duration (us)"),
        TraceLoggingInt64(profiler_record->time_span_, "Time Stamp (us)"),
        TraceLoggingString(profiler_record->event_name_, "Event Name"),
        TraceLoggingInt32(profiler_record->process_id_, "Process ID"),
        TraceLoggingInt32(profiler_record->thread_id_, "Thread ID"),
        TraceLoggingString(profiler_record->op_name_, "Operator Name"),
        TraceLoggingString(profiler_record->execution_provider_, "Execution Provider"));
  } else {
    TraceLoggingWrite(
        winml_trace_logging_provider,
        "OnnxRuntimeProfiling",
        TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_LOTUS_PROFILING),
        TraceLoggingLevel(WINEVENT_LEVEL_VERBOSE),
        TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
        TraceLoggingString(profiler_record->category_name_, "Category"),
        TraceLoggingInt64(profiler_record->duration_, "Duration (us)"),
        TraceLoggingInt64(profiler_record->time_span_, "Time Stamp (us)"),
        TraceLoggingString(profiler_record->event_name_, "Event Name"),
        TraceLoggingInt32(profiler_record->process_id_, "Process ID"),
        TraceLoggingInt32(profiler_record->thread_id_, "Thread ID"));
  }
}

static void OnSuspending(winrt::Windows::Foundation::IInspectable const& sender, winrt::Windows::ApplicationModel::SuspendingEventArgs const& args) {
  telemetry_helper.LogWinMLSuspended();
}

void OnnxruntimeEnvironment::RegisterSuspendHandler() {
  try {
    auto suspend_event_handler = winrt::Windows::Foundation::EventHandler<winrt::Windows::ApplicationModel::SuspendingEventArgs>(&OnSuspending);
    suspend_token_ = winrt::Windows::ApplicationModel::Core::CoreApplication::Suspending(suspend_event_handler);
  } catch (...) {
  }  //Catch in case CoreApplication cannot be found for non-UWP executions
}

OnnxruntimeEnvironment::OnnxruntimeEnvironment(const OrtApi* ort_api) : ort_env_(nullptr, nullptr) {
  OrtEnv* ort_env = nullptr;
  THROW_IF_NOT_OK_MSG(ort_api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env),
                      ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->SetLanguageProjection(ort_env, OrtLanguageProjection::ORT_PROJECTION_WINML), ort_api);
  ort_env_ = UniqueOrtEnv(ort_env, ort_api->ReleaseEnv);
  // Configure the environment with the winml logger
  auto winml_adapter_api = GetVersionedWinmlAdapterApi(ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->EnvConfigureCustomLoggerAndProfiler(ort_env_.get(),
                                                                             &WinmlOrtLoggingCallback, &WinmlOrtProfileEventCallback, nullptr,
                                                                             OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env),
                      ort_api);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->OverrideSchema(), ort_api);

  // Register suspend handler for UWP applications
  RegisterSuspendHandler();
}

OnnxruntimeEnvironment::~OnnxruntimeEnvironment() {
  if (suspend_token_) {
    winrt::Windows::ApplicationModel::Core::CoreApplication::Suspending(suspend_token_);
  }
}

HRESULT OnnxruntimeEnvironment::GetOrtEnvironment(_Out_ OrtEnv** ort_env) {
  *ort_env = ort_env_.get();
  return S_OK;
}

HRESULT OnnxruntimeEnvironment::EnableDebugOutput(bool is_enabled) {
  debug_output_ = is_enabled;
  return S_OK;
}
