// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>

#include <ILogConfiguration.hpp>
#include <NullObjects.hpp>

#include "core/platform/telemetry_1ds.h"
#include "core/platform/telemetry_1ds_platform.h"

namespace onnxruntime {
namespace {

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s (Win32 error: %lu)\n", message, GetLastError());
    std::fflush(stderr);
    // Also fail reliably from a DLL process-detach destructor.
    TerminateProcess(GetCurrentProcess(), EXIT_FAILURE);
    std::abort();
  }
}

}  // namespace

class OneDsTelemetryTestPeer {
 public:
  static void Install(Microsoft::Applications::Events::ILogManager& manager) {
    Check(OneDsTelemetry::global_register_count_ == 1, "Expected exactly one telemetry instance");
    Check(!OneDsTelemetry::log_manager_ && !OneDsTelemetry::config_, "Real uploader must be disabled");
    OneDsTelemetry::config_ = std::make_unique<Microsoft::Applications::Events::ILogConfiguration>();
    OneDsTelemetry::log_manager_ = &manager;
  }

  static bool IsClean() {
    return OneDsTelemetry::global_register_count_ == 0 &&
           !OneDsTelemetry::log_manager_ && !OneDsTelemetry::config_;
  }
};

namespace {

class MockLogManager : public Microsoft::Applications::Events::NullLogManager {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MockLogManager);

  explicit MockLogManager(bool pending) : callback_(CreateEventW(nullptr, TRUE, !pending, nullptr)) {
    Check(callback_ != nullptr, "CreateEventW failed");
  }

  ~MockLogManager() override { CloseHandle(callback_); }

  Microsoft::Applications::Events::status_t Flush() override {
    ++flush_count;
    return Microsoft::Applications::Events::STATUS_SUCCESS;
  }

  void FlushAndTeardown() override {
    Check(flush_count == 1, "Expected Flush before FlushAndTeardown");
    ++teardown_count;
    // A pending callback cannot arrive after ExitProcess has killed the other
    // threads. Bound the mock's wait so a regression does not leave a stuck DLL.
    Check(WaitForSingleObject(callback_, 2000) == WAIT_OBJECT_0, "Timed out draining the pending uploader");
  }

  int flush_count = 0;
  int teardown_count = 0;

 private:
  HANDLE callback_;
};

class TelemetryLifetime {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TelemetryLifetime);

  TelemetryLifetime(bool process_exit, HANDLE verified)
      : manager_(process_exit), process_exit_(process_exit), verified_(verified) {
    telemetry_.emplace();
    OneDsTelemetryTestPeer::Install(manager_);
  }

  ~TelemetryLifetime() {
    if (telemetry_) {
      Destroy();
    }
  }

  void Destroy() {
    Check(telemetry_internal::IsDllShutdownInProgress() == process_exit_, "Unexpected DLL shutdown state");
    telemetry_.reset();
    const int expected_calls = process_exit_ ? 0 : 1;
    Check(manager_.flush_count == expected_calls, "Unexpected Flush count");
    Check(manager_.teardown_count == expected_calls, "Unexpected FlushAndTeardown count");
    if (!process_exit_) {
      Check(OneDsTelemetryTestPeer::IsClean(), "Telemetry state was not cleaned up");
    }
    Check(SetEvent(verified_) != FALSE, "Could not signal verified destruction");
  }

 private:
  MockLogManager manager_;
  std::optional<OneDsTelemetry> telemetry_;
  bool process_exit_;
  HANDLE verified_;
};

}  // namespace
}  // namespace onnxruntime

extern "C" void OrtTestTelemetryShutdown(bool process_exit, bool destroy_now, HANDLE verified) {
  // A DLL-owned static exercises CRT destruction during real Windows shutdown.
  static onnxruntime::TelemetryLifetime lifetime(process_exit, verified);
  if (destroy_now) {
    lifetime.Destroy();
  }
}
