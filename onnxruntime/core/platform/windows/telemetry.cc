// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/telemetry.h"
#include "core/common/version.h"

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

namespace onnxruntime {

namespace {
TRACELOGGING_DEFINE_PROVIDER(telemetry_provider_handle, "Microsoft.ML.ONNXRuntime",
                             // {3a26b1ff-7484-7484-7484-15261f42614d}
                             (0x3a26b1ff, 0x7484, 0x7484, 0x74, 0x84, 0x15, 0x26, 0x1f, 0x42, 0x61, 0x4d),
                             TraceLoggingOptionMicrosoftTelemetry());
}  // namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

WindowsTelemetry::WindowsTelemetry() {
  HRESULT hr = TraceLoggingRegister(telemetry_provider_handle);
  if (SUCCEEDED(hr)) {
    register_succeeded_ = true;
  }
}

WindowsTelemetry::~WindowsTelemetry() {
  if (register_succeeded_) {
    TraceLoggingUnregister(telemetry_provider_handle);
    register_succeeded_ = false;
  }
}

void WindowsTelemetry::LogProcessInfo(const std::string& runtimeVersion, bool isRedist) {
  if (!register_succeeded_)
    return;

  TraceLoggingWrite(telemetry_provider_handle,
                    "ProcessInfo",
                    TraceLoggingBool(true, "UTCReplace_AppSessionGuid"),
                    TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage),
                    // Telemetry info
                    TraceLoggingKeyword(1),
                    TraceLoggingUInt8(0, "schemaVersion"),
                    TraceLoggingString(ONNXRUNTIME_VERSION_STRING, "runtimeVersion"),
                    TraceLoggingBool(true, "isRedist"),
                    TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES));

}

void WindowsTelemetry::LogSessionCreation(uint32_t sessionId, int64_t irVersion, const std::string& modelProducerName,
                                          const std::string& modelProducerVersion, const std::string& modelDomain,
                                          const std::vector<std::string>& modelOpsetImports, uint32_t modelPrecision,
                                          const std::string& modelGraphName, const std::string& modelGraphVersion,
                                          const std::unordered_map<std::string, std::string>& modelMetaData,
                                          bool modelFromStream, const std::string& executionProviders) {
}

void WindowsTelemetry::LogRuntimeError(uint32_t sessionId, const common::Status& status, const char* file,
                                       const char* function, uint32_t line) {
}

void WindowsTelemetry::LogRuntimePerf(uint32_t sessionId, uint32_t runTotalTimeMs) {
}

}  // namespace onnxruntime
