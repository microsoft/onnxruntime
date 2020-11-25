// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "controller.h"

Controller::Controller() : cleanup_group_(CreateThreadpoolCleanupGroup()), event_(CreateOnnxRuntimeEvent()) {
  InitializeThreadpoolEnvironment(&env_);
  SetThreadpoolCallbackPool(&env_, nullptr);
  SetThreadpoolCallbackCleanupGroup(&env_, cleanup_group_, nullptr);
}

Controller::~Controller() noexcept { free(errmsg_); }

bool Controller::RunAsync(_Inout_ ONNXRUNTIME_CALLBACK_FUNCTION callback, _In_ void* data) {
  std::lock_guard<std::mutex> g(m_);
  if (state_ == State::RUNNING) {
    ::CreateAndSubmitThreadpoolWork(callback, data, &env_);
    return true;
  }
  return false;
}

std::string Controller::Wait() {
  WaitAndCloseEvent(event_);
  CloseThreadpoolCleanupGroupMembers(cleanup_group_, errmsg_ == nullptr ? FALSE : TRUE, nullptr);
  CloseThreadpoolCleanupGroup(cleanup_group_);
  return errmsg_ == nullptr ? std::string() : errmsg_;
}

void Controller::SetFailBit(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci, _In_ const char* err_msg) {
  std::lock_guard<std::mutex> g(m_);
  if (state_ == State::RUNNING || state_ == State::SHUTDOWN) {
    state_ = State::STOPPED;
    is_running_ = false;
    errmsg_ = my_strdup(err_msg);
    ::OnnxRuntimeSetEventWhenCallbackReturns(pci, event_);
  }
}

bool Controller::SetEof(ONNXRUNTIME_CALLBACK_INSTANCE pci) {
  std::lock_guard<std::mutex> g(m_);
  if (state_ == State::RUNNING) {
    state_ = State::SHUTDOWN;
    ::OnnxRuntimeSetEventWhenCallbackReturns(pci, event_);
    return true;
  }
  return false;
}
