// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
// Copied with modifications from
// https://www.justsoftwaresolutions.co.uk/threading/implementing-a-thread-safe-queue-using-condition-variables.html
// Modifications Copyright (c) Microsoft.

#pragma once

#include <deque>
#include <mutex>
#include "core/platform/ort_mutex.h"
#include <condition_variable>

namespace onnxruntime {
template <typename ElemType>
class ConcurrentQueue {
 public:
  enum class Status {
    kSuccess,
    kTimedout
  };

  ConcurrentQueue() {}

  bool Empty() const {
    std::lock_guard<OrtMutex> lk(mu_);
    return queue_.empty();
  }

  Status WaitAndPop(int wait_millis, ElemType& popped_value) {
    std::unique_lock<OrtMutex> lk(mu_);
    while (queue_.empty()) {
      auto rc = cv_.wait_for(lk, wait_millis * std::chrono::milliseconds(1));
      if (rc == std::cv_status::timeout) {
        return Status::kTimedout;
      }
    }
    popped_value = queue_.front();
    queue_.pop_front();
    return Status::kSuccess;
  }

  void Push(const ElemType& value) {
    {
      std::lock_guard<OrtMutex> lk(mu_);
      queue_.push_back(value);
    }
    cv_.notify_all();
  }

  void Push(ElemType&& value) {
    {
      std::lock_guard<OrtMutex> lk(mu_);
      queue_.push_back(std::move(value));
    }
    cv_.notify_all();
  }

 private:
  OrtMutex mu_;
  OrtCondVar cv_;
  std::deque<ElemType> queue_;
};
}  // namespace onnxruntime
