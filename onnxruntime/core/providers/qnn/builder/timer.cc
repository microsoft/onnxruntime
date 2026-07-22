// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "timer.h"

void Timer::DeInitialize() {
  std::unique_lock<std::mutex> lk(mtx_);
  is_timer_deinit_ = true;
  cv_.notify_all();
  lk.unlock();
  if (bkg_thread_.joinable()) {
    bkg_thread_.join();
  }
}

Timer::~Timer() { this->DeInitialize(); }

void Timer::BkgTimer() {
  {
    std::unique_lock<std::mutex> lk(mtx_);
    thread_status_ = threadState::IDLE;
    cv_.notify_all();
  }
  while (true) {
    std::unique_lock<std::mutex> lk(mtx_);

    if (thread_status_ == threadState::IDLE) {
      cv_.wait(lk, [&]() {
        return is_timer_launched_ || is_timer_stopped_ || is_timer_deinit_;
      });
    }

    if (is_timer_deinit_) {
      thread_status_ = threadState::DEINIT;
      is_timer_deinit_ = false;
      return;
    }

    if (is_timer_stopped_) {
      thread_status_ = threadState::IDLE;
      is_timer_stopped_ = false;
      cv_.notify_all();
    }

    if (thread_status_ == threadState::LAUNCH) {
      bool isElapsed = !cv_.wait_until(lk, end_time_, [&]() {
        return is_timer_stopped_ || is_timer_deinit_;
      });
      if (isElapsed) {
        thread_status_ = threadState::CALLING;
        lk.unlock();
        timeout_fn_(timeout_arg_);
        lk.lock();
        thread_status_ = threadState::IDLE;
      }
      is_timer_launched_ = false;
    }
  }
}

bool Timer::Initialize(std::function<void(void*)> callbackFn, void* callbackArg) {
  std::unique_lock<std::mutex> lk(mtx_);
  timeout_arg_ = callbackArg;
  timeout_fn_ = callbackFn;
  bkg_thread_ = std::thread(&Timer::BkgTimer, this);
  cv_.wait(lk, [&] { return thread_status_ == threadState::IDLE; });
  return true;
}

void Timer::AbortTimer() {
  std::unique_lock<std::mutex> lk(mtx_);
  is_timer_stopped_ = true;
  cv_.notify_all();
  cv_.wait(lk, [&] { return thread_status_ == threadState::IDLE; });
}

bool Timer::TimerInUse() {
  return thread_status_ == threadState::LAUNCH;
}
