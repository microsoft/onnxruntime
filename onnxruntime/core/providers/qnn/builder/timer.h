// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <iostream>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <atomic>

class Timer {
 public:
  enum class threadState {
    IDLE,     // Timer is created
    LAUNCH,   // Timer starts counting down
    CALLING,  // Callback function is called
    DEINIT    // Timer is deinit
  };
  // constructor
  Timer() = default;
  // destructor
  ~Timer();

  template <class T_Rep, class T_Period>
  bool RemainingDuration(std::chrono::duration<T_Rep, T_Period>& duration) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (thread_status_ == threadState::LAUNCH) {
      duration = std::chrono::duration_cast<std::chrono::duration<T_Rep, T_Period>>(end_time_ - std::chrono::steady_clock::now());
      return true;
    } else if (thread_status_ == threadState::CALLING || thread_status_ == threadState::IDLE) {
      duration = std::chrono::duration<T_Rep, T_Period>::zero();
      return true;
    } else {
      duration = std::chrono::duration<T_Rep, T_Period>::zero();
      return false;
    }
  }

  template <class T_Rep, class T_Period>
  bool Launch(const std::chrono::duration<T_Rep, T_Period>& timeoutVal) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (thread_status_ != threadState::IDLE) {
      return false;
    }
    end_time_ = std::chrono::steady_clock::now() + timeoutVal;
    thread_status_ = threadState::LAUNCH;
    is_timer_launched_ = true;
    cv_.notify_all();
    return true;
  }

  bool Initialize(std::function<void(void*)> callbackFn, void* callbackArg);
  void DeInitialize();
  void AbortTimer();

  bool TimerInUse();

 private:
  std::thread bkg_thread_;
  void BkgTimer();
  std::mutex mtx_;
  std::condition_variable cv_;
  std::function<void(void*)> timeout_fn_;
  void* timeout_arg_{nullptr};
  std::atomic<threadState> thread_status_{threadState::DEINIT};
  std::chrono::time_point<std::chrono::steady_clock> end_time_;
  std::atomic<bool> is_timer_stopped_ = false;
  std::atomic<bool> is_timer_deinit_ = false;
  std::atomic<bool> is_timer_launched_ = false;
};
