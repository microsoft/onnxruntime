// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "sync_api.h"
#include <atomic>
#include <mutex>

class Controller {
 private:
  PTP_CLEANUP_GROUP const cleanup_group_;
  TP_CALLBACK_ENVIRON env_;
  ONNXRUNTIME_EVENT event_;
  std::atomic<bool> is_running_ = true;
  std::mutex m_;
  enum class State { RUNNING, SHUTDOWN, STOPPED } state_ = State::RUNNING;
  char* errmsg_ = nullptr;

 public:
  Controller();
  ~Controller() noexcept;
  Controller(const Controller&) = delete;
  Controller& operator=(const Controller&) = delete;
  // return true if SetFailBit has not been called
  bool IsRunning() const { return is_running_; }

  void SetFailBit(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci, _In_ const char* err_msg);
  bool SetEof(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci);

  // Wait the state becoming stopped, and all the submitted work has been finished(or cancelled if error happened)
  std::string Wait();
  bool RunAsync(_Inout_ ONNXRUNTIME_CALLBACK_FUNCTION callback, _In_ void* data);
};