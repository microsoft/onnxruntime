// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32
#include <Windows.h>
#include <mutex>
namespace onnxruntime {
// Q: Why OrtMutex is better than std::mutex
// A: OrtMutex supports static initialization but std::mutex doesn't. Static initialization helps us prevent the "static
// initialization order problem".

// Q: Why std::mutex can't make it?
// A: VC runtime has to support Windows XP at ABI level. But we don't have such requirement.

// Q: Is OrtMutex faster than std::mutex?
// A: Sure

class OrtMutex {
 private:
  SRWLOCK data_ = SRWLOCK_INIT;

 public:
  constexpr OrtMutex() = default;
  // SRW locks do not need to be explicitly destroyed.
  ~OrtMutex() = default;
  OrtMutex(const OrtMutex&) = delete;
  OrtMutex& operator=(const OrtMutex&) = delete;
  void lock() { AcquireSRWLockExclusive(native_handle()); }
  bool try_lock() noexcept { return TryAcquireSRWLockExclusive(native_handle()) == TRUE; }
  void unlock() noexcept { ReleaseSRWLockExclusive(native_handle()); }
  using native_handle_type = SRWLOCK*;

  __forceinline native_handle_type native_handle() { return &data_; }
};

class OrtCondVar {
  CONDITION_VARIABLE native_cv_object = CONDITION_VARIABLE_INIT;

 public:
  constexpr OrtCondVar() noexcept = default;
  ~OrtCondVar() = default;

  OrtCondVar(const OrtCondVar&) = delete;
  OrtCondVar& operator=(const OrtCondVar&) = delete;

  void notify_one() noexcept { WakeConditionVariable(&native_cv_object); }
  void notify_all() noexcept { WakeAllConditionVariable(&native_cv_object); }

  void wait(std::unique_lock<OrtMutex>& lk) {
    if (SleepConditionVariableSRW(&native_cv_object, lk.mutex()->native_handle(), INFINITE, 0) != TRUE) {
      std::terminate();
    }
  }
  template <class _Predicate>
  void wait(std::unique_lock<OrtMutex>& __lk, _Predicate __pred);

  /**
   * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
   * cv_status::no_timeout.
   * @param cond_mutex A unique_lock<OrtMutex> object.
   * @param rel_time A chrono::duration object that specifies the amount of time before the thread wakes up.
   * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
   * cv_status::no_timeout
   */
  template <class Rep, class Period>
  std::cv_status wait_for(std::unique_lock<OrtMutex>& cond_mutex, const std::chrono::duration<Rep, Period>& rel_time);
  using native_handle_type = CONDITION_VARIABLE*;

  native_handle_type native_handle() { return &native_cv_object; }

 private:
  void timed_wait_impl(std::unique_lock<OrtMutex>& __lk,
                       std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

template <class _Predicate>
void OrtCondVar::wait(std::unique_lock<OrtMutex>& __lk, _Predicate __pred) {
  while (!__pred()) wait(__lk);
}

template <class Rep, class Period>
std::cv_status OrtCondVar::wait_for(std::unique_lock<OrtMutex>& cond_mutex,
                                    const std::chrono::duration<Rep, Period>& rel_time) {
  // TODO: is it possible to use nsync_from_time_point_ ?
  using namespace std::chrono;
  if (rel_time <= duration<Rep, Period>::zero())
    return std::cv_status::timeout;
  using SystemTimePointFloat = time_point<system_clock, duration<long double, std::nano> >;
  using SystemTimePoint = time_point<system_clock, nanoseconds>;
  SystemTimePointFloat max_time = SystemTimePoint::max();
  steady_clock::time_point steady_now = steady_clock::now();
  system_clock::time_point system_now = system_clock::now();
  if (max_time - rel_time > system_now) {
    nanoseconds remain = duration_cast<nanoseconds>(rel_time);
    if (remain < rel_time)
      ++remain;
    timed_wait_impl(cond_mutex, system_now + remain);
  } else
    timed_wait_impl(cond_mutex, SystemTimePoint::max());
  return steady_clock::now() - steady_now < rel_time ? std::cv_status::no_timeout : std::cv_status::timeout;
}
}  // namespace onnxruntime
#elif defined(ENABLE_ORT_WASM)
#include <mutex>
#include <condition_variable>
namespace onnxruntime {
  using OrtMutex = std::mutex;
  using OrtCondVar = std::condition_variable;
}
#else
#include "nsync.h"
#include <mutex>               //for unique_lock
#include <condition_variable>  //for cv_status
namespace onnxruntime {

class OrtMutex {
  nsync::nsync_mu data_ = NSYNC_MU_INIT;

 public:
  constexpr OrtMutex() = default;
  ~OrtMutex() = default;
  OrtMutex(const OrtMutex&) = delete;
  OrtMutex& operator=(const OrtMutex&) = delete;

  void lock() { nsync::nsync_mu_lock(&data_); }
  bool try_lock() noexcept { return nsync::nsync_mu_trylock(&data_) == 0; }
  void unlock() noexcept { nsync::nsync_mu_unlock(&data_); }

  using native_handle_type = nsync::nsync_mu*;
  native_handle_type native_handle() { return &data_; }
};

class OrtCondVar {
  nsync::nsync_cv native_cv_object = NSYNC_CV_INIT;

 public:
  constexpr OrtCondVar() noexcept = default;

  ~OrtCondVar() = default;
  OrtCondVar(const OrtCondVar&) = delete;
  OrtCondVar& operator=(const OrtCondVar&) = delete;

  void notify_one() noexcept { nsync::nsync_cv_signal(&native_cv_object); }
  void notify_all() noexcept { nsync::nsync_cv_broadcast(&native_cv_object); }

  void wait(std::unique_lock<OrtMutex>& lk);
  template <class _Predicate>
  void wait(std::unique_lock<OrtMutex>& __lk, _Predicate __pred);

  /**
   * returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
   * cv_status::no_timeout.
   * @param cond_mutex A unique_lock<OrtMutex> object.
   * @param rel_time A chrono::duration object that specifies the amount of time before the thread wakes up.
   * @return returns cv_status::timeout if the wait terminates when Rel_time has elapsed. Otherwise, the method returns
   * cv_status::no_timeout
   */
  template <class Rep, class Period>
  std::cv_status wait_for(std::unique_lock<OrtMutex>& cond_mutex, const std::chrono::duration<Rep, Period>& rel_time);
  using native_handle_type = nsync::nsync_cv*;
  native_handle_type native_handle() { return &native_cv_object; }

 private:
  void timed_wait_impl(std::unique_lock<OrtMutex>& __lk,
                       std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>);
};

template <class _Predicate>
void OrtCondVar::wait(std::unique_lock<OrtMutex>& __lk, _Predicate __pred) {
  while (!__pred()) wait(__lk);
}

template <class Rep, class Period>
std::cv_status OrtCondVar::wait_for(std::unique_lock<OrtMutex>& cond_mutex,
                                    const std::chrono::duration<Rep, Period>& rel_time) {
  // TODO: is it possible to use nsync_from_time_point_ ?
  using namespace std::chrono;
  if (rel_time <= duration<Rep, Period>::zero())
    return std::cv_status::timeout;
  using SystemTimePointFloat = time_point<system_clock, duration<long double, std::nano> >;
  using SystemTimePoint = time_point<system_clock, nanoseconds>;
  SystemTimePointFloat max_time = SystemTimePoint::max();
  steady_clock::time_point steady_now = steady_clock::now();
  system_clock::time_point system_now = system_clock::now();
  if (max_time - rel_time > system_now) {
    nanoseconds remain = duration_cast<nanoseconds>(rel_time);
    if (remain < rel_time)
      ++remain;
    timed_wait_impl(cond_mutex, system_now + remain);
  } else
    timed_wait_impl(cond_mutex, SystemTimePoint::max());
  return steady_clock::now() - steady_now < rel_time ? std::cv_status::no_timeout : std::cv_status::timeout;
}
};  // namespace onnxruntime
#endif
