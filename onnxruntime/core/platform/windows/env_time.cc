/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env_time.h"

#include <time.h>
#include <windows.h>
#include <chrono>
#include <numeric>
#include <algorithm>

namespace onnxruntime {

namespace {

class WindowsEnvTime : public EnvTime {
 public:
#if WINVER >= _WIN32_WINNT_WIN8
  WindowsEnvTime() : GetSystemTimePreciseAsFileTime_(GetSystemTimePreciseAsFileTime) {}
#else
  WindowsEnvTime() : GetSystemTimePreciseAsFileTime_(NULL) {
    // GetSystemTimePreciseAsFileTime function is only available in
    // Windows >= 8. For that reason, we try to look it up in
    // kernel32.dll at runtime and use an alternative option if the function
    // is not available.
    HMODULE module = GetModuleHandleW(L"kernel32.dll");
    if (module != NULL) {
      auto func = (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
          module, "GetSystemTimePreciseAsFileTime");
      GetSystemTimePreciseAsFileTime_ = func;
    }
  }
#endif

  uint64_t NowMicros() override {
    if (GetSystemTimePreciseAsFileTime_ != NULL) {
      // GetSystemTimePreciseAsFileTime function is only available in latest
      // versions of Windows, so we need to check for its existence here.
      // All std::chrono clocks on Windows proved to return
      // values that may repeat, which is not good enough for some uses.
      constexpr int64_t kUnixEpochStartTicks = 116444736000000000i64;
      constexpr int64_t kFtToMicroSec = 10;

      // This interface needs to return system time and not
      // just any microseconds because it is often used as an argument
      // to TimedWait() on condition variable
      FILETIME system_time;
      GetSystemTimePreciseAsFileTime_(&system_time);

      LARGE_INTEGER li;
      li.LowPart = system_time.dwLowDateTime;
      li.HighPart = system_time.dwHighDateTime;
      // Subtract unix epoch start
      li.QuadPart -= kUnixEpochStartTicks;
      // Convert to microsecs
      li.QuadPart /= kFtToMicroSec;
      return li.QuadPart;
    }
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch())
        .count();
  }

  void SleepForMicroseconds(int64_t micros) { Sleep(static_cast<DWORD>(micros) / 1000); }

 private:
  typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
  FnGetSystemTimePreciseAsFileTime GetSystemTimePreciseAsFileTime_;
};

}  // namespace

EnvTime* EnvTime::Default() {
  static WindowsEnvTime default_time_env;
  return &default_time_env;
}

bool GetMonotonicTimeCounter(TIME_SPEC* value) {
  static_assert(sizeof(LARGE_INTEGER) == sizeof(TIME_SPEC), "type mismatch");
  return QueryPerformanceCounter((LARGE_INTEGER*)value) != 0;
}

static INIT_ONCE g_InitOnce = INIT_ONCE_STATIC_INIT;
static LARGE_INTEGER freq;

BOOL CALLBACK InitHandleFunction(
    PINIT_ONCE,
    PVOID,
    PVOID*) {
  return QueryPerformanceFrequency(&freq);
}

void SetTimeSpecToZero(TIME_SPEC* value) {
  *value = 0;
}

void AccumulateTimeSpec(TIME_SPEC* base, TIME_SPEC* start, TIME_SPEC* end) {
  *base += std::max<TIME_SPEC>(0, *end - *start);
}

//Return the interval in seconds.
//If the function fails, the return value is zero
double TimeSpecToSeconds(TIME_SPEC* value) {
  BOOL initState = InitOnceExecuteOnce(&g_InitOnce, InitHandleFunction, nullptr, nullptr);
  if (!initState) return 0;
  return *value / (double)freq.QuadPart;
}

}  // namespace onnxruntime
