// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/telemetry.h"
#include <winapifamily.h>
#include <cwchar>
#include <cstdint>
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#include <shellapi.h>
#endif
#include <winsvc.h>
#include <mutex>
#include <string>
#include <vector>
#include "core/common/logging/logging.h"
#include "core/platform/telemetry_environment.h"
#include "core/platform/telemetry_redaction.h"
#include "onnxruntime_config.h"

// ETW includes
// need space after Windows.h to prevent clang-format re-ordering breaking the build.
// TraceLoggingProvider.h must follow Windows.h
#include <Windows.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26440)  // Warning C26440 from TRACELOGGING_DEFINE_PROVIDER
#endif

#include <TraceLoggingProvider.h>
#include <evntrace.h>
#include <winmeta.h>

// Seems this workaround can be dropped when we drop support for VS2017 toolchains
// https://developercommunity.visualstudio.com/content/problem/85934/traceloggingproviderh-is-incompatible-with-utf-8.html
#ifdef _TlgPragmaUtf8Begin
#undef _TlgPragmaUtf8Begin
#define _TlgPragmaUtf8Begin
#endif

#ifdef _TlgPragmaUtf8End
#undef _TlgPragmaUtf8End
#define _TlgPragmaUtf8End
#endif

// Different versions of TraceLoggingProvider.h contain different macro variable names for the utf8 begin and end,
// and we need to cover the lower case version as well.
#ifdef _tlgPragmaUtf8Begin
#undef _tlgPragmaUtf8Begin
#define _tlgPragmaUtf8Begin
#endif

#ifdef _tlgPragmaUtf8End
#undef _tlgPragmaUtf8End
#define _tlgPragmaUtf8End
#endif

namespace onnxruntime {

namespace {
TRACELOGGING_DEFINE_PROVIDER(telemetry_provider_handle, "Microsoft.ML.ONNXRuntime",
                             // {3a26b1ff-7484-7484-7484-15261f42614d}
                             (0x3a26b1ff, 0x7484, 0x7484, 0x74, 0x84, 0x15, 0x26, 0x1f, 0x42, 0x61, 0x4d),
                             TraceLoggingOptionMicrosoftTelemetry());

std::string GetCpuModel() {
  HKEY key{};
  if (::RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0,
                      KEY_READ,
                      &key) != ERROR_SUCCESS) {
    return "unknown";
  }

  char cpu_model[256]{};
  DWORD value_type = REG_SZ;
  DWORD size = sizeof(cpu_model);
  const LSTATUS status = ::RegQueryValueExA(key,
                                            "ProcessorNameString",
                                            nullptr,
                                            &value_type,
                                            reinterpret_cast<LPBYTE>(cpu_model),
                                            &size);
  ::RegCloseKey(key);

  if (status != ERROR_SUCCESS || value_type != REG_SZ || size == 0) {
    return "unknown";
  }

  cpu_model[sizeof(cpu_model) - 1] = '\0';
  return cpu_model[0] != '\0' ? std::string(cpu_model) : std::string("unknown");
}

uint32_t GetProcessorCount() {
  SYSTEM_INFO system_info{};
  ::GetSystemInfo(&system_info);
  return static_cast<uint32_t>(system_info.dwNumberOfProcessors);
}

uint64_t GetTotalMemoryMB() {
  MEMORYSTATUSEX memory_status{};
  memory_status.dwLength = sizeof(memory_status);
  if (::GlobalMemoryStatusEx(&memory_status) == 0) {
    return 0;
  }

  return memory_status.ullTotalPhys / (1024 * 1024);
}

std::string ConvertWideStringToUtf8(std::wstring_view wide) {
  wide = telemetry_detail::TelemetryWideStringView(wide);
  if (wide.empty())
    return {};

  const UINT code_page = CP_UTF8;
  const DWORD flags = 0;
  LPCWCH const src = wide.data();
  const int src_len = static_cast<int>(wide.size());
  int utf8_length = ::WideCharToMultiByte(code_page, flags, src, src_len, nullptr, 0, nullptr, nullptr);
  if (utf8_length == 0)
    return {};

  std::string utf8(utf8_length, '\0');
  if (::WideCharToMultiByte(code_page, flags, src, src_len, utf8.data(), utf8_length, nullptr, nullptr) == 0)
    return {};

  return utf8;
}

// Parse the command line for -s (service name) and -k (service group) arguments.
// These are svchost.exe conventions, so this only applies when the host process image is
// svchost.exe; for any other process these flags are unrelated and their values are not collected.
std::string GetServiceNamesFromCommandLine() {
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
  // The -s/-k service-name convention is specific to svchost.exe. Restrict command-line parsing to
  // svchost so a same-named flag in an unrelated host process does not leak its argument value.
  wchar_t module_path[MAX_PATH];
  DWORD module_path_len = ::GetModuleFileNameW(nullptr, module_path, MAX_PATH);
  if (module_path_len == 0 || module_path_len >= MAX_PATH)
    return {};
  const wchar_t* image_name = ::wcsrchr(module_path, L'\\');
  image_name = (image_name != nullptr) ? image_name + 1 : module_path;
  if (_wcsicmp(image_name, L"svchost.exe") != 0)
    return {};

  LPCWSTR cmd_line = ::GetCommandLineW();
  if (cmd_line == nullptr)
    return {};

  int argc = 0;
  LPWSTR* argv = ::CommandLineToArgvW(cmd_line, &argc);
  if (argv == nullptr)
    return {};

  std::string aggregated;
  bool first = true;
  size_t count = 0;
  for (int i = 0; i < argc - 1; ++i) {
    if ((_wcsicmp(argv[i], L"-s") == 0 || _wcsicmp(argv[i], L"-k") == 0)) {
      if (count++ == telemetry_detail::kMaxTelemetryCollectionEntries) break;
      if (!first) {
        if (!telemetry_detail::AppendTelemetryString(aggregated, ",")) break;
      }
      const std::string value = ConvertWideStringToUtf8(telemetry_detail::TelemetryWideStringView(argv[i + 1]));
      if (!telemetry_detail::AppendTelemetryString(aggregated, value) ||
          aggregated.size() == kMaxTelemetryStringLength) break;
      first = false;
      ++i;  // skip the value we just consumed
    }
  }

  ::LocalFree(argv);
  return aggregated;
#else
  // CommandLineToArgvW lives in shell32 and is only available on the desktop partition; the
  // svchost -s/-k service-name convention does not apply on non-desktop Windows (UWP/GDK).
  return {};
#endif
}

std::string GetServiceNamesForCurrentProcess() {
  static std::once_flag once_flag;
  static std::string service_names;

  std::call_once(once_flag, [] {
    SC_HANDLE service_manager = ::OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ENUMERATE_SERVICE);
    if (service_manager == nullptr) {
      service_names = GetServiceNamesFromCommandLine();
      return;
    }

    DWORD bytes_needed = 0;
    DWORD services_returned = 0;
    DWORD resume_handle = 0;
    if (!::EnumServicesStatusExW(service_manager, SC_ENUM_PROCESS_INFO, SERVICE_WIN32, SERVICE_ACTIVE, nullptr, 0, &bytes_needed,
                                 &services_returned, &resume_handle, nullptr) &&
        ::GetLastError() != ERROR_MORE_DATA) {
      ::CloseServiceHandle(service_manager);
      service_names = GetServiceNamesFromCommandLine();
      return;
    }

    // EnumServicesStatusEx supports at most a 256 KiB enumeration buffer.
    if (bytes_needed == 0 || bytes_needed > 256 * 1024) {
      ::CloseServiceHandle(service_manager);
      service_names = GetServiceNamesFromCommandLine();
      return;
    }

    std::vector<uint8_t> buffer(bytes_needed);
    auto* services = reinterpret_cast<ENUM_SERVICE_STATUS_PROCESSW*>(buffer.data());
    services_returned = 0;
    resume_handle = 0;
    if (!::EnumServicesStatusExW(service_manager, SC_ENUM_PROCESS_INFO, SERVICE_WIN32, SERVICE_ACTIVE, reinterpret_cast<LPBYTE>(services),
                                 bytes_needed, &bytes_needed, &services_returned, &resume_handle, nullptr)) {
      ::CloseServiceHandle(service_manager);
      service_names = GetServiceNamesFromCommandLine();
      return;
    }

    DWORD current_pid = ::GetCurrentProcessId();
    std::string aggregated;
    bool first = true;
    size_t count = 0;
    for (DWORD i = 0; i < services_returned; ++i) {
      if (services[i].ServiceStatusProcess.dwProcessId == current_pid) {
        if (count++ == telemetry_detail::kMaxTelemetryCollectionEntries) break;
        if (!first) {
          if (!telemetry_detail::AppendTelemetryString(aggregated, ",")) break;
        }
        const std::string value =
            ConvertWideStringToUtf8(telemetry_detail::TelemetryWideStringView(services[i].lpServiceName));
        if (!telemetry_detail::AppendTelemetryString(aggregated, value) ||
            aggregated.size() == kMaxTelemetryStringLength) break;
        first = false;
      }
    }

    ::CloseServiceHandle(service_manager);

    service_names = std::move(aggregated);
    if (service_names.empty()) {
      service_names = GetServiceNamesFromCommandLine();
    }
  });

  return service_names;
}
}  // namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifndef ORT_CALLER_FRAMEWORK
#define ORT_CALLER_FRAMEWORK ""
#endif

std::mutex WindowsTelemetry::mutex_;
std::mutex WindowsTelemetry::provider_change_mutex_;
uint32_t WindowsTelemetry::global_register_count_ = 0;
bool WindowsTelemetry::enabled_ = true;
uint32_t WindowsTelemetry::projection_ = 0;
UCHAR WindowsTelemetry::level_ = 0;
UINT64 WindowsTelemetry::keyword_ = 0;
std::vector<const WindowsTelemetry::EtwInternalCallback*> WindowsTelemetry::callbacks_;
std::mutex WindowsTelemetry::callbacks_mutex_;

WindowsTelemetry::WindowsTelemetry() {
  std::lock_guard<std::mutex> lock(mutex_);
  // ORT_RUNNING_UNIT_TESTS is an internal hard-suppression signal, unlike the user-facing
  // non-Windows environment opt-out. Do not register the ETW provider in test processes.
  if (IsRunningUnitTests()) {
    enabled_ = false;
    return;
  }
  if (global_register_count_ == 0) {
    // TraceLoggingRegister is fancy in that you can only register once GLOBALLY for the whole process
    HRESULT hr = TraceLoggingRegisterEx(telemetry_provider_handle, ORT_TL_EtwEnableCallback, nullptr);
    if (SUCCEEDED(hr)) {
      global_register_count_ += 1;
    }
  }
}

WindowsTelemetry::~WindowsTelemetry() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (global_register_count_ > 0) {
    global_register_count_ -= 1;
    if (global_register_count_ == 0) {
      TraceLoggingUnregister(telemetry_provider_handle);
    }
  }

  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.clear();
}

bool WindowsTelemetry::IsEnabled() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return enabled_;
}

UCHAR WindowsTelemetry::Level() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return level_;
}

UINT64 WindowsTelemetry::Keyword() const {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  return keyword_;
}

// HRESULT WindowsTelemetry::Status() {
//     return etw_status_;
// }

void WindowsTelemetry::RegisterInternalCallback(const EtwInternalCallback& callback) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  callbacks_.push_back(&callback);
}

void WindowsTelemetry::UnregisterInternalCallback(const EtwInternalCallback& callback) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  auto new_end = std::remove_if(callbacks_.begin(), callbacks_.end(),
                                [&callback](const EtwInternalCallback* ptr) {
                                  return ptr == &callback;
                                });
  callbacks_.erase(new_end, callbacks_.end());
}

void NTAPI WindowsTelemetry::ORT_TL_EtwEnableCallback(
    _In_ LPCGUID SourceId,
    _In_ ULONG IsEnabled,
    _In_ UCHAR Level,
    _In_ ULONGLONG MatchAnyKeyword,
    _In_ ULONGLONG MatchAllKeyword,
    _In_opt_ PEVENT_FILTER_DESCRIPTOR FilterData,
    _In_opt_ PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock(provider_change_mutex_);
  enabled_ = (IsEnabled != 0);
  level_ = Level;
  keyword_ = MatchAnyKeyword;

  InvokeCallbacks(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
}

void WindowsTelemetry::InvokeCallbacks(LPCGUID SourceId, ULONG IsEnabled, UCHAR Level, ULONGLONG MatchAnyKeyword,
                                       ULONGLONG MatchAllKeyword, PEVENT_FILTER_DESCRIPTOR FilterData,
                                       PVOID CallbackContext) {
  std::lock_guard<std::mutex> lock_callbacks(callbacks_mutex_);
  for (const auto& callback : callbacks_) {
    (*callback)(SourceId, IsEnabled, Level, MatchAnyKeyword, MatchAllKeyword, FilterData, CallbackContext);
  }
}

void WindowsTelemetry::EnableTelemetryEvents() const {
  enabled_ = true;
}

void WindowsTelemetry::DisableTelemetryEvents() const {
  enabled_ = false;
}

void WindowsTelemetry::SetLanguageProjection(uint32_t projection) const {
  projection_ = projection;
}

void WindowsTelemetry::LogProcessInfo() const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  static std::atomic<bool> process_info_logged;

  // did we already log the process info?  we only need to log it once
  if (process_info_logged.exchange(true))
    return;
  bool isRedist = true;
#if BUILD_INBOX
  isRedist = false;
#endif
  const std::string service_names = GetServiceNamesForCurrentProcess();
  const std::string cpu_model = GetCpuModel();
  const uint32_t processor_count = GetProcessorCount();
  const uint64_t total_memory_mb = GetTotalMemoryMB();
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "ProcessInfo",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(cpu_model.c_str()), "cpuModel"),
                    TraceLoggingUInt32(processor_count, "processorCount"),
                    TraceLoggingUInt64(total_memory_mb, "totalMemoryMB"),
                    TraceLoggingBool(IsDebuggerPresent(), "isDebuggerAttached"),
                    TraceLoggingBool(isRedist, "isRedist"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"),
                    TraceLoggingString(strings.Utf8(service_names.c_str()), "serviceNames"));

  process_info_logged = true;
}

void WindowsTelemetry::LogSessionCreationStart(uint32_t session_id) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "SessionCreationStart",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogEvaluationStop(uint32_t session_id) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "EvaluationStop",
                    TraceLoggingUInt32(session_id, "sessionId"));
}

void WindowsTelemetry::LogEvaluationStart(uint32_t session_id) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "EvaluationStart",
                    TraceLoggingUInt32(session_id, "sessionId"));
}

void WindowsTelemetry::LogSessionCreation(uint32_t session_id, int64_t ir_version, const std::string& model_producer_name,
                                          const std::string& model_producer_version, const std::string& model_domain,
                                          const std::unordered_map<std::string, int>& domain_to_version_map,
                                          const std::string& model_file_name,
                                          const std::string& model_graph_name,
                                          const std::string& model_weight_type,
                                          const std::string& model_graph_hash,
                                          const std::string& model_weight_hash,
                                          const std::unordered_map<std::string, std::string>& model_metadata,
                                          const std::string& loaded_from, const std::vector<std::string>& execution_provider_ids,
                                          const std::string& hardware_device_types,
                                          const std::string& hardware_vendor_ids,
                                          const std::string& ep_versions,
                                          bool use_fp16, bool captureState) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string domain_to_version_string = telemetry_detail::FormatTelemetryMap(domain_to_version_map);
  const std::string model_metadata_string = telemetry_detail::FormatTelemetryMap(model_metadata);
  const std::string execution_provider_string = telemetry_detail::JoinTelemetryStrings(execution_provider_ids);

  const std::string service_names = GetServiceNamesForCurrentProcess();
  // Difference is MeasureEvent & isCaptureState, but keep in sync otherwise
  if (!captureState) {
    telemetry_detail::TelemetryStrings strings;
    TraceLoggingWrite(telemetry_provider_handle,
                      "SessionCreation",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      // Telemetry info
                      // schemaVersion 1: added hardwareDeviceTypes and hardwareVendorIds
                      // schemaVersion 2: added executionProviderVersions
                      TraceLoggingUInt8(2, "schemaVersion"),
                      TraceLoggingUInt32(session_id, "sessionId"),
                      TraceLoggingInt64(ir_version, "irVersion"),
                      TraceLoggingUInt32(projection_, "OrtProgrammingProjection"),
                      TraceLoggingString(strings.Utf8(model_producer_name.c_str()), "modelProducerName"),
                      TraceLoggingString(strings.Utf8(model_producer_version.c_str()), "modelProducerVersion"),
                      TraceLoggingString(strings.Utf8(model_domain.c_str()), "modelDomain"),
                      TraceLoggingBool(use_fp16, "usefp16"),
                      TraceLoggingString(strings.Utf8(domain_to_version_string.c_str()), "domainToVersionMap"),
                      TraceLoggingString(strings.Utf8(model_file_name.c_str()), "modelFileName"),
                      TraceLoggingString(strings.Utf8(model_graph_name.c_str()), "modelGraphName"),
                      TraceLoggingString(strings.Utf8(model_weight_type.c_str()), "modelWeightType"),
                      TraceLoggingString(strings.Utf8(model_graph_hash.c_str()), "modelGraphHash"),
                      TraceLoggingString(strings.Utf8(model_weight_hash.c_str()), "modelWeightHash"),
                      TraceLoggingString(strings.Utf8(model_metadata_string.c_str()), "modelMetaData"),
                      TraceLoggingString(strings.Utf8(loaded_from.c_str()), "loadedFrom"),
                      TraceLoggingString(strings.Utf8(execution_provider_string.c_str()), "executionProviderIds"),
                      TraceLoggingString(strings.Utf8(hardware_device_types.c_str()), "hardwareDeviceTypes"),
                      TraceLoggingString(strings.Utf8(hardware_vendor_ids.c_str()), "hardwareVendorIds"),
                      TraceLoggingString(strings.Utf8(ep_versions.c_str()), "executionProviderVersions"),
                      TraceLoggingString(strings.Utf8(service_names.c_str()), "serviceNames"),
                      TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
  } else {
    telemetry_detail::TelemetryStrings strings;
    TraceLoggingWrite(telemetry_provider_handle,
                      "SessionCreation_CaptureState",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                      // Not a measure event
                      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      // Telemetry info
                      // schemaVersion 1: added hardwareDeviceTypes and hardwareVendorIds
                      // schemaVersion 2: added executionProviderVersions
                      TraceLoggingUInt8(2, "schemaVersion"),
                      TraceLoggingUInt32(session_id, "sessionId"),
                      TraceLoggingInt64(ir_version, "irVersion"),
                      TraceLoggingUInt32(projection_, "OrtProgrammingProjection"),
                      TraceLoggingString(strings.Utf8(model_producer_name.c_str()), "modelProducerName"),
                      TraceLoggingString(strings.Utf8(model_producer_version.c_str()), "modelProducerVersion"),
                      TraceLoggingString(strings.Utf8(model_domain.c_str()), "modelDomain"),
                      TraceLoggingBool(use_fp16, "usefp16"),
                      TraceLoggingString(strings.Utf8(domain_to_version_string.c_str()), "domainToVersionMap"),
                      TraceLoggingString(strings.Utf8(model_file_name.c_str()), "modelFileName"),
                      TraceLoggingString(strings.Utf8(model_graph_name.c_str()), "modelGraphName"),
                      TraceLoggingString(strings.Utf8(model_weight_type.c_str()), "modelWeightType"),
                      TraceLoggingString(strings.Utf8(model_graph_hash.c_str()), "modelGraphHash"),
                      TraceLoggingString(strings.Utf8(model_weight_hash.c_str()), "modelWeightHash"),
                      TraceLoggingString(strings.Utf8(model_metadata_string.c_str()), "modelMetaData"),
                      TraceLoggingString(strings.Utf8(loaded_from.c_str()), "loadedFrom"),
                      TraceLoggingString(strings.Utf8(execution_provider_string.c_str()), "executionProviderIds"),
                      TraceLoggingString(strings.Utf8(hardware_device_types.c_str()), "hardwareDeviceTypes"),
                      TraceLoggingString(strings.Utf8(hardware_vendor_ids.c_str()), "hardwareVendorIds"),
                      TraceLoggingString(strings.Utf8(ep_versions.c_str()), "executionProviderVersions"),
                      TraceLoggingString(strings.Utf8(service_names.c_str()), "serviceNames"),
                      TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
  }
}

void WindowsTelemetry::LogCompileModelStart(uint32_t session_id,
                                            const std::string& input_source,
                                            const std::string& output_target,
                                            uint32_t flags,
                                            int graph_optimization_level,
                                            bool embed_ep_context,
                                            bool has_external_initializers_file,
                                            const std::vector<std::string>& execution_provider_ids) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string execution_provider_string = telemetry_detail::JoinTelemetryStrings(execution_provider_ids);

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "CompileModelStart",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingString(strings.Utf8(input_source.c_str()), "inputSource"),
                    TraceLoggingString(strings.Utf8(output_target.c_str()), "outputTarget"),
                    TraceLoggingUInt32(flags, "flags"),
                    TraceLoggingInt32(graph_optimization_level, "graphOptimizationLevel"),
                    TraceLoggingBool(embed_ep_context, "embedEpContext"),
                    TraceLoggingBool(has_external_initializers_file, "hasExternalInitializersFile"),
                    TraceLoggingString(strings.Utf8(execution_provider_string.c_str()), "executionProviderIds"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogCompileModelComplete(uint32_t session_id,
                                               bool success,
                                               uint32_t error_code,
                                               uint32_t error_category,
                                               const std::string& error_message) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string scrubbed_error = ScrubStringForTelemetry(error_message);
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "CompileModelComplete",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingBool(success, "success"),
                    TraceLoggingUInt32(error_code, "errorCode"),
                    TraceLoggingUInt32(error_category, "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file,
                                       const char* function, uint32_t line) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string scrubbed_error = ScrubStringForTelemetry(status.ErrorMessage());
  std::string_view file_view = telemetry_detail::TelemetryStringView(file, telemetry_detail::kMaxTelemetryPathBytes);
  if (const size_t slash = file_view.find_last_of("/\\"); slash != std::string_view::npos) {
    file_view.remove_prefix(slash + 1);
  }
  const std::string scrubbed_file = ScrubStringForTelemetry(file_view);
#ifdef _WIN32
  HRESULT hr = common::StatusCodeToHRESULT(static_cast<common::StatusCode>(status.Code()));
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RuntimeError",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
                    // Telemetry info
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingHResult(hr, "hResult"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(scrubbed_file.c_str()), "file"),
                    TraceLoggingString(strings.Utf8(function), "function"),
                    TraceLoggingInt32(line, "line"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
#else
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RuntimeError",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
                    // Telemetry info
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(scrubbed_file.c_str()), "file"),
                    TraceLoggingString(strings.Utf8(function), "function"),
                    TraceLoggingInt32(line, "line"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
#endif
}

void WindowsTelemetry::LogRuntimeInferenceError(uint32_t session_id, const common::Status& status,
                                                const std::string& ep_versions,
                                                const std::string& ep_device_types) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string scrubbed_error = ScrubStringForTelemetry(status.ErrorMessage());
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RuntimeInferenceError",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_ERROR),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(ep_versions.c_str()), "executionProviderVersions"),
                    TraceLoggingString(strings.Utf8(ep_device_types.c_str()), "executionProviderDeviceTypes"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogRuntimePerf(uint32_t session_id, uint32_t total_runs_since_last, int64_t total_run_duration_since_last,
                                      const std::unordered_map<int64_t, long long>& duration_per_batch_size) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string total_duration_per_batch_size =
      telemetry_detail::FormatTelemetryMap(duration_per_batch_size, ", ", ": ");

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RuntimePerf",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    // Telemetry info
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingUInt32(total_runs_since_last, "totalRuns"),
                    TraceLoggingInt64(total_run_duration_since_last, "totalRunDuration"),
                    TraceLoggingString(strings.Utf8(total_duration_per_batch_size.c_str()), "totalRunDurationPerBatchSize"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogEpDeviceUsage(uint32_t session_id,
                                        const std::string& ep_type,
                                        const std::string& hardware_device_type,
                                        uint32_t hardware_vendor_id,
                                        uint32_t hardware_device_id,
                                        const std::string& hardware_vendor,
                                        const std::string& ep_vendor,
                                        const std::string& ep_version,
                                        int assigned_node_count,
                                        uint32_t total_runs_since_last,
                                        int64_t total_run_duration_since_last) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "EpDeviceUsage",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    // schemaVersion 1: added epVersion, runtimeVersion
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingString(strings.Utf8(ep_type.c_str()), "executionProviderType"),
                    TraceLoggingString(strings.Utf8(hardware_device_type.c_str()), "hardwareDeviceType"),
                    TraceLoggingUInt32(hardware_vendor_id, "hardwareVendorId"),
                    TraceLoggingUInt32(hardware_device_id, "hardwareDeviceId"),
                    TraceLoggingString(strings.Utf8(hardware_vendor.c_str()), "hardwareVendor"),
                    TraceLoggingString(strings.Utf8(ep_vendor.c_str()), "epVendor"),
                    TraceLoggingString(strings.Utf8(ep_version.c_str()), "epVersion"),
                    TraceLoggingInt32(assigned_node_count, "assignedNodeCount"),
                    TraceLoggingUInt32(total_runs_since_last, "totalRunsSinceLast"),
                    TraceLoggingInt64(total_run_duration_since_last, "totalRunDurationSinceLast"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogExecutionProviderEvent(LUID* adapterLuid) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "ExecutionProviderEvent",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    // Telemetry info
                    TraceLoggingUInt32(adapterLuid->LowPart, "adapterLuidLowPart"),
                    TraceLoggingUInt32(adapterLuid->HighPart, "adapterLuidHighPart"));
}

void WindowsTelemetry::LogDriverInfoEvent(const std::string_view device_class, const std::wstring_view& driver_names, const std::wstring_view& driver_versions) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "DriverInfo",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingString(strings.Utf8(device_class), "deviceClass"),
                    TraceLoggingWideString(strings.Wide(driver_names), "driverNames"),
                    TraceLoggingWideString(strings.Wide(driver_versions), "driverVersions"));
}

void WindowsTelemetry::LogAutoEpSelection(uint32_t session_id, const std::string& selection_policy,
                                          const std::vector<std::string>& requested_execution_provider_ids,
                                          const std::vector<std::string>& available_execution_provider_ids) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string requested_execution_provider_string =
      telemetry_detail::JoinTelemetryStrings(requested_execution_provider_ids);
  const std::string available_execution_provider_string =
      telemetry_detail::JoinTelemetryStrings(available_execution_provider_ids);

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "EpAutoSelection",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingString(strings.Utf8(selection_policy.c_str()), "selectionPolicy"),
                    TraceLoggingString(strings.Utf8(requested_execution_provider_string.c_str()), "requestedExecutionProviderIds"),
                    TraceLoggingString(strings.Utf8(available_execution_provider_string.c_str()), "availableExecutionProviderIds"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogProviderOptions(const std::string& provider_id, const std::string& provider_options_string, bool captureState) const {
  LogLocalProviderOptions(provider_id, provider_options_string, captureState);
}

void WindowsTelemetry::LogLocalProviderOptions(const std::string& provider_id,
                                               const std::string& provider_options_string,
                                               bool capture_state) {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  // Difference is MeasureEvent & isCaptureState, but keep in sync otherwise
  if (!capture_state) {
    telemetry_detail::TelemetryStrings strings;
    TraceLoggingWrite(telemetry_provider_handle,
                      "ProviderOptions",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                      TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      // Telemetry info
                      TraceLoggingUInt8(0, "schemaVersion"),
                      TraceLoggingString(strings.Utf8(provider_id.c_str()), "providerId"),
                      TraceLoggingString(strings.Utf8(provider_options_string.c_str()), "providerOptions"),
                      TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
  } else {
    telemetry_detail::TelemetryStrings strings;
    TraceLoggingWrite(telemetry_provider_handle,
                      "ProviderOptions_CaptureState",
                      TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                      TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                      // Not a measure event
                      TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
                      TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                      // Telemetry info
                      TraceLoggingUInt8(0, "schemaVersion"),
                      TraceLoggingString(strings.Utf8(provider_id.c_str()), "providerId"),
                      TraceLoggingString(strings.Utf8(provider_options_string.c_str()), "providerOptions"),
                      TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
  }
}

void WindowsTelemetry::LogModelLoadStart(uint32_t session_id) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "ModelLoadStart",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogModelLoadEnd(uint32_t session_id, const common::Status& status,
                                       int64_t duration_us) const {
  ORT_UNUSED_PARAMETER(duration_us);
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string scrubbed_error = status.IsOK() ? std::string() : ScrubStringForTelemetry(status.ErrorMessage());
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "ModelLoadEnd",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingBool(status.IsOK(), "isSuccess"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogSessionCreationEnd(uint32_t session_id,
                                             const common::Status& status,
                                             int64_t duration_us) const {
  ORT_UNUSED_PARAMETER(duration_us);
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string scrubbed_error = status.IsOK() ? std::string() : ScrubStringForTelemetry(status.ErrorMessage());
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "SessionCreationEnd",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingUInt32(session_id, "sessionId"),
                    TraceLoggingBool(status.IsOK(), "isSuccess"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogRegisterEpLibraryWithLibPath(const std::string& registration_name,
                                                       const std::string& lib_path) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RegisterEpLibraryWithLibPath",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingString(strings.Utf8(registration_name.c_str()), "registrationName"),
                    TraceLoggingString(strings.Utf8(lib_path.c_str()), "libPath"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogRegisterEpLibraryStart(const std::string& registration_name) const {
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RegisterEpLibraryStart",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(1, "schemaVersion"),
                    TraceLoggingString(strings.Utf8(registration_name.c_str()), "registrationName"),
                    TraceLoggingString(strings.Utf8(ORT_VERSION), "runtimeVersion"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

void WindowsTelemetry::LogRegisterEpLibraryEnd(const std::string& registration_name,
                                               const common::Status& status,
                                               int64_t duration_us) const {
  ORT_UNUSED_PARAMETER(duration_us);
  if (global_register_count_ == 0 || enabled_ == false)
    return;

  const std::string scrubbed_error = status.IsOK() ? std::string() : ScrubStringForTelemetry(status.ErrorMessage());
  telemetry_detail::TelemetryStrings strings;
  TraceLoggingWrite(telemetry_provider_handle,
                    "RegisterEpLibraryEnd",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
                    TraceLoggingLevel(WINEVENT_LEVEL_INFO),
                    // Telemetry info
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingString(strings.Utf8(registration_name.c_str()), "registrationName"),
                    TraceLoggingBool(status.IsOK(), "isSuccess"),
                    TraceLoggingUInt32(status.Code(), "errorCode"),
                    TraceLoggingUInt32(status.Category(), "errorCategory"),
                    TraceLoggingString(strings.Utf8(scrubbed_error.c_str()), "errorMessage"),
                    TraceLoggingString(strings.Utf8(ORT_CALLER_FRAMEWORK), "frameworkName"));
}

}  // namespace onnxruntime
