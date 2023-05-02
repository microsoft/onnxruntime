// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// EtwSink.h must come before the windows includes
#include "core/platform/windows/logging/etw_sink.h"

#ifdef ETW_TRACE_LOGGING_SUPPORTED

// STL includes
#include <exception>

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

// See: https://developercommunity.visualstudio.com/content/problem/85934/traceloggingproviderh-is-incompatible-with-utf-8.html
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
namespace logging {

namespace {
TRACELOGGING_DEFINE_PROVIDER(etw_provider_handle, "ONNXRuntimeTraceLoggingProvider",
                             // {929DD115-1ECB-4CB5-B060-EBD4983C421D}
                             (0x929dd115, 0x1ecb, 0x4cb5, 0xb0, 0x60, 0xeb, 0xd4, 0x98, 0x3c, 0x42, 0x1d));
}  // namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

// Class to unregister ETW provider at shutdown.
// We expect one static instance to be created for the lifetime of the program.
class EtwRegistrationManager {
 public:
  static EtwRegistrationManager& Register() {
    const HRESULT etw_status = ::TraceLoggingRegister(etw_provider_handle);

    if (FAILED(etw_status)) {
      ORT_THROW("ETW registration failed. Logging will be broken: " + std::to_string(etw_status));
    }

    // return an instance that is just used to unregister as the program exits
    static EtwRegistrationManager instance(etw_status);
    return instance;
  }

  const HRESULT Status() const noexcept {
    return etw_status_;
  }

  ~EtwRegistrationManager() {
    ::TraceLoggingUnregister(etw_provider_handle);
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(EtwRegistrationManager);

  EtwRegistrationManager(const HRESULT status) noexcept : etw_status_{status} {}
  const HRESULT etw_status_;
};

void EtwSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  UNREFERENCED_PARAMETER(timestamp);

  // register on first usage
  static EtwRegistrationManager& etw_manager = EtwRegistrationManager::Register();

  // do something (not that meaningful) with etw_manager so it doesn't get optimized out
  // as we want an instance around to do the unregister
  if (FAILED(etw_manager.Status())) {
    return;
  }

  // Do we want to output Verbose level messages via ETW at any point it time?
  // TODO: Validate if this filtering makes sense.
  if (message.Severity() <= Severity::kVERBOSE || message.DataType() == DataType::USER) {
    return;
  }

  // NOTE: Theoretically we could create an interface for all the ETW system interactions so we can separate
  // out those from the logic in this class so it is more testable.
  // Right now the logic is trivial, so that effort isn't worth it.

  // TraceLoggingWrite requires (painfully) a compile time constant for the TraceLoggingLevel,
  // forcing us to use an ugly macro for the call.
#define ETW_EVENT_NAME "ONNXRuntimeLogEvent"
#define TRACE_LOG_WRITE(level)                                                             \
  TraceLoggingWrite(etw_provider_handle, ETW_EVENT_NAME, TraceLoggingLevel(level),         \
                    TraceLoggingString(logger_id.c_str(), "logger"),                       \
                    TraceLoggingString(message.Category(), "category"),                    \
                    TraceLoggingString(message.Location().ToString().c_str(), "location"), \
                    TraceLoggingString(message.Message().c_str(), "message"))

  const auto severity{message.Severity()};

  GSL_SUPPRESS(bounds)
  GSL_SUPPRESS(type) {
    switch (severity) {
      case Severity::kVERBOSE:
        TRACE_LOG_WRITE(TRACE_LEVEL_VERBOSE);
        break;
      case Severity::kINFO:
        TRACE_LOG_WRITE(TRACE_LEVEL_INFORMATION);
        break;
      case Severity::kWARNING:
        TRACE_LOG_WRITE(TRACE_LEVEL_WARNING);
        break;
      case Severity::kERROR:
        TRACE_LOG_WRITE(TRACE_LEVEL_ERROR);
        break;
      case Severity::kFATAL:
        TRACE_LOG_WRITE(TRACE_LEVEL_CRITICAL);
        break;
      default:
        ORT_THROW("Unexpected Severity of " + std::to_string(static_cast<int>(severity)));
    }
  }

#undef ETW_EVENT_NAME
#undef TRACE_LOG_WRITE
}
}  // namespace logging
}  // namespace onnxruntime

#endif  // ETW_TRACE_LOGGING_SUPPORTED
