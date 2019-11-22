// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "core/common/logging/isink.h"
#include "WinMLProfiler.h"
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include "inc/WinMLAdapter.h"

#pragma warning(push)
#pragma warning(disable : 4505)

namespace Windows {
namespace AI {
namespace MachineLearning {
class CWinMLLogSink : public onnxruntime::logging::ISink {
 public:
  CWinMLLogSink() {
  }
  static void EnableDebugOutput() {
    debug_output_ = true;
    OutputDebugStringW(L"Windows.AI.MachineLearning: Debug Output Enabled \r\n");
  }
  void SendProfileEvent(onnxruntime::profiling::EventRecord& event_record) const;
  void SendImpl(const onnxruntime::logging::Timestamp& timestamp, const std::string& logger_id, const onnxruntime::logging::Capture& message);

 private:
  static bool debug_output_;
};
// TODO: a bug in ORT requires a logging manager.  This function registers a static singleton logger as "default"
inline onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  // create a CLog based default logging manager
  static std::string default_logger_id{"Default"};
  static onnxruntime::logging::LoggingManager default_logging_manager{
      std::unique_ptr<onnxruntime::logging::ISink>{new CWinMLLogSink()},
      onnxruntime::logging::Severity::kVERBOSE,
      false,
      onnxruntime::logging::LoggingManager::InstanceType::Default,
      &default_logger_id,
      MAXINT32};

  return default_logging_manager;
}

static void OnSuspending(winrt::Windows::Foundation::IInspectable const& sender, winrt::Windows::ApplicationModel::SuspendingEventArgs const& args) {
#ifdef LAYERING_DONE
    if (!profiler.IsStillReset())  //If profiler is still reset, then don't log RuntimePerf
  {
    telemetry_helper.LogRuntimePerf(profiler, true);
  }
#endif
}

class LotusEnvironment {
 public:
  LotusEnvironment() {
    const HRESULT etw_status = TraceLoggingRegister(winml_trace_logging_provider);
    if (FAILED(etw_status)) {
      throw std::runtime_error("WinML TraceLogging registration failed. Logging will be broken: " + std::to_string(etw_status));
    }

    // TODO: Do we need to call this or just define the method?
    default_logging_manager_ = &DefaultLoggingManager();

    if (!onnxruntime::Environment::Create(lotus_environment_).IsOK()) {
      throw winrt::hresult_error(E_FAIL);
    }

    auto allocatorMap = onnxruntime::DeviceAllocatorRegistry::Instance().AllRegistrations();
    if (allocatorMap.find("Cpu") == allocatorMap.end()) {
      onnxruntime::DeviceAllocatorRegistry::Instance().RegisterDeviceAllocator(
          "Cpu",
          [](int) { return std::make_unique<onnxruntime::CPUAllocator>(); },
          std::numeric_limits<size_t>::max());
    }
    RegisterSuspendHandler();
  }

  ~LotusEnvironment() {
    TraceLoggingUnregister(winml_trace_logging_provider);
    if (suspend_token_) {
      winrt::Windows::ApplicationModel::Core::CoreApplication::Suspending(suspend_token_);
    }
  }

  const onnxruntime::logging::Logger* GetDefaultLogger() {
    return &default_logging_manager_->DefaultLogger();
  }

 private:
  void RegisterSuspendHandler() {
    try {
        suspend_token_ = winrt::Windows::ApplicationModel::Core::CoreApplication::Suspending(
            winrt::Windows::Foundation::EventHandler<winrt::Windows::ApplicationModel::SuspendingEventArgs>(&OnSuspending));
    } catch (...) {}//Catch in case CoreApplication cannot be found for non-UWP executions
  }

  std::unique_ptr<onnxruntime::Environment> lotus_environment_;
  winrt::event_token suspend_token_;
  onnxruntime::logging::LoggingManager* default_logging_manager_;
};

namespace ExecutionProviders {
__declspec(selectany) const char* CPUExecutionProvider = "CPUExecutionProvider";
}

}  // namespace MachineLearning
}  // namespace AI
}  // namespace Windows

#pragma warning(pop)