// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "sync_api.h"

#include <mutex>

template <typename T>
class FixedCountFinishCallbackImpl {
 private:
  //remain tasks
  int s_;
  std::mutex m_;
  ONNXRUNTIME_EVENT finish_event_;
  bool failed = false;
  std::vector<std::shared_ptr<T>> results_;

 public:
  FixedCountFinishCallbackImpl(const FixedCountFinishCallbackImpl&) = delete;
  FixedCountFinishCallbackImpl& operator=(const FixedCountFinishCallbackImpl&) = delete;

  const std::vector<std::shared_ptr<T>>& getResults() const {
    return results_;
  }

  FixedCountFinishCallbackImpl(int s) : s_(s), results_(s) {
    ONNXRUNTIME_ENFORCE(CreateOnnxRuntimeEvent(&finish_event_).IsOK());
  }

  ~FixedCountFinishCallbackImpl() {
    if (finish_event_) OrtCloseEvent(finish_event_);
  }

  ::onnxruntime::common::Status fail(ONNXRUNTIME_CALLBACK_INSTANCE pci) {
    {
      std::lock_guard<std::mutex> g(m_);
      failed = true;
      s_ = 0;  //fail earlier
    }
    return OnnxRuntimeSetEventWhenCallbackReturns(pci, finish_event_);
  }

  ::onnxruntime::common::Status onFinished(size_t task_index, std::shared_ptr<T> result, ONNXRUNTIME_CALLBACK_INSTANCE pci) {
    int v;
    {
      std::lock_guard<std::mutex> g(m_);
      v = --s_;
      results_.at(task_index) = result;
    }
    if (v == 0) {
      return OnnxRuntimeSetEventWhenCallbackReturns(pci, finish_event_);
    }
    return ::onnxruntime::common::Status::OK();
  }

  bool shouldStop() {
    std::lock_guard<std::mutex> g(m_);
    return failed;
  }
  //this function can only be invoked once
  bool wait() {
    ONNXRUNTIME_ENFORCE(WaitAndCloseEvent(finish_event_).IsOK());
    {
      std::lock_guard<std::mutex> g(m_);
      finish_event_ = nullptr;
      return !failed;
    }
  }
};
