// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/ort_mutex.h"
#include <assert.h>
#include <stdexcept>
#include <system_error>
#include <sstream>

namespace onnxruntime {
#ifndef USE_NSYNC
OrtMutex::~OrtMutex() {
  pthread_mutex_destroy(&data_);
}
#endif

void
OrtMutex::lock() {
#ifndef USE_NSYNC
  int ec = pthread_mutex_lock(&data_);
  if (ec)
    throw std::system_error(ec, std::system_category(), "mutex lock failed");
#else
  nsync::nsync_mu_lock(&data_);
#endif
}

bool
OrtMutex::try_lock()
noexcept {
#ifndef USE_NSYNC
  return pthread_mutex_trylock(&data_) == 0;
#else
  return nsync::nsync_mu_trylock(&data_) == 0;
#endif
}

void
OrtMutex::unlock()
noexcept
{
#ifdef USE_NSYNC
  nsync::nsync_mu_unlock(&data_);
#else
int ec = pthread_mutex_unlock(&data_);
(void) ec;
//Don't throw!
assert(ec== 0);
#endif
}

#ifndef USE_NSYNC
OrtCondVar::~OrtCondVar() {
  pthread_cond_destroy(&native_cv_object);
}
#endif

void OrtCondVar::notify_one() noexcept {
#ifdef USE_NSYNC
  nsync::nsync_cv_signal(&native_cv_object);
#else
  pthread_cond_signal(&native_cv_object);
#endif
}

void OrtCondVar::notify_all() noexcept {
#ifdef USE_NSYNC
  nsync::nsync_cv_broadcast(&native_cv_object);
#else
  pthread_cond_broadcast(&native_cv_object);
#endif
}

void OrtCondVar::wait(std::unique_lock<OrtMutex>& lk) {
  if (!lk.owns_lock())
    throw std::runtime_error("OrtCondVar wait failed: mutex not locked");
#ifdef USE_NSYNC
  nsync::nsync_cv_wait(&native_cv_object, lk.mutex()->native_handle());
#else
  int ec = pthread_cond_wait(&native_cv_object, lk.mutex()->native_handle());
  if (ec) {
    std::ostringstream oss;
    oss << "OrtCondVar wait failed, error code=" << ec;
    throw std::runtime_error(oss.str());
  }
#endif
}

void OrtCondVar::timed_wait_impl(std::unique_lock<OrtMutex>& lk,
                                 std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> tp) {
  using namespace std::chrono;
#ifndef NDEBUG
  if (!lk.owns_lock())
    throw std::runtime_error("condition_variable::timed wait: mutex not locked");
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
#ifdef USE_NSYNC
  nsync::nsync_cv_wait_with_deadline(&native_cv_object, lk.mutex()->native_handle(), abs_deadline, nullptr);
#else
  int ec = pthread_cond_timedwait(&native_cv_object, lk.mutex()->native_handle(), &abs_deadline);
  if (ec != 0 && ec != ETIMEDOUT) {
    std::ostringstream oss;
    oss << "OrtCondVar timed_wait failed, error code=" << ec;
    throw std::runtime_error(oss.str());
  }
#endif
}

}  // namespace onnxruntime