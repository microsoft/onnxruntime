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
#else
#ifdef USE_NSYNC
#include "nsync.h"
#include <mutex>               //for unique_lock
#include <condition_variable>  //for cv_status
#else
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#endif
namespace onnxruntime {

class OrtMutex {
#ifdef USE_NSYNC
  nsync::nsync_mu data_ = NSYNC_MU_INIT;
#else
  pthread_mutex_t data_ = PTHREAD_MUTEX_INITIALIZER;
#endif

 public:
  constexpr OrtMutex() = default;
#ifdef USE_NSYNC
  ~OrtMutex() = default;
#else
  ~OrtMutex();
#endif

  OrtMutex(const OrtMutex&) = delete;
  OrtMutex& operator=(const OrtMutex&) = delete;

  void lock();
  bool try_lock() noexcept;
  void unlock() noexcept;

#ifdef USE_NSYNC
  using native_handle_type = nsync::nsync_mu*;
#else
  using native_handle_type = pthread_mutex_t*;
#endif
  native_handle_type native_handle() { return &data_; }
};

class OrtCondVar {
#ifdef USE_NSYNC
  nsync::nsync_cv native_cv_object = NSYNC_CV_INIT;
#else
  pthread_cond_t native_cv_object = PTHREAD_COND_INITIALIZER;
#endif
 public:
  constexpr OrtCondVar() noexcept = default;

#ifdef USE_NSYNC
  ~OrtCondVar() = default;
#else
  ~OrtCondVar();
#endif

  OrtCondVar(const OrtCondVar&) = delete;
  OrtCondVar& operator=(const OrtCondVar&) = delete;

  void notify_one() noexcept;
  void notify_all() noexcept;

  void wait(std::unique_lock<OrtMutex>& __lk);
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
#ifdef USE_NSYNC
  using native_handle_type = nsync::nsync_cv*;
#else
  using native_handle_type = pthread_cond_t*;
#endif

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