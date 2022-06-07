// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include <assert.h>
#include <stdexcept>
#include <sstream>

namespace onnxruntime {
void OrtCondVar::timed_wait_impl(std::unique_lock<OrtMutex>& lk,
                                 std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp) {
  using namespace std::chrono;
#ifndef NDEBUG
  if (!lk.owns_lock())
    ORT_THROW("condition_variable::timed wait: mutex not locked");
#endif
  nanoseconds d = tp.time_since_epoch();
  timespec abs_deadline;
  seconds s = duration_cast<seconds>(d);
  using ts_sec = decltype(abs_deadline.tv_sec);
  constexpr ts_sec ts_sec_max = std::numeric_limits<ts_sec>::max();
  if (s.count() < ts_sec_max) {
    abs_deadline.tv_sec = static_cast<ts_sec>(s.count());
    abs_deadline.tv_nsec = static_cast<decltype(abs_deadline.tv_nsec)>((d - s).count());
  } else {
    abs_deadline.tv_sec = ts_sec_max;
    abs_deadline.tv_nsec = 999999999;
  }
  nsync::nsync_cv_wait_with_deadline(&native_cv_object, lk.mutex()->native_handle(), abs_deadline, nullptr);
}

void OrtCondVar::wait(std::unique_lock<OrtMutex>& lk) {
#ifndef NDEBUG
  if (!lk.owns_lock()) {
    ORT_THROW("OrtCondVar wait failed: mutex not locked");
  }
#endif
  nsync::nsync_cv_wait(&native_cv_object, lk.mutex()->native_handle());
}

}  // namespace onnxruntime