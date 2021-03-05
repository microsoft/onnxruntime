// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include <assert.h>
#include <stdexcept>
#include <system_error>
#include <sstream>

namespace onnxruntime {
void OrtCondVar::timed_wait_impl(std::unique_lock<OrtMutex>& lk,
                                 std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp) {
  auto dwMilliseconds = static_cast<DWORD>(tp.time_since_epoch().count());
  if (SleepConditionVariableSRW(&native_cv_object, lk.mutex()->native_handle(), dwMilliseconds, 0) != TRUE) {
    if (GetLastError() != ERROR_TIMEOUT) {
      std::terminate();
    }
  }
}

}  // namespace onnxruntime