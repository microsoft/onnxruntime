// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"

namespace onnxruntime {
struct OrtCallback {
  void (*f)(void* param) noexcept;
  void* param;
};

/**
 *  f will be freed in this call
 */
void OrtRunCallback(OrtCallback* f) noexcept;

/**
 * Invokes the contained OrtCallback with operator()(T).
 * Useful for something like a std::unique_ptr<> deleter.
 */
struct OrtCallbackInvoker {
  OrtCallbackInvoker() noexcept
      : callback{nullptr, nullptr} {}

  OrtCallbackInvoker(OrtCallback callback_to_invoke) noexcept
      : callback(callback_to_invoke) {}

  OrtCallback callback;

  template <typename T>
  void operator()(T) noexcept {
    if (callback.f) {
      callback.f(callback.param);
    }
  }
};

/**
 * Invokes the contained OrtCallback upon destruction or being assigned to.
 */
class ScopedOrtCallbackInvoker {
 public:
  explicit ScopedOrtCallbackInvoker(OrtCallback callback) noexcept
      : callback_(callback) {}

  ScopedOrtCallbackInvoker(ScopedOrtCallbackInvoker&& other) noexcept
      : callback_(other.callback_) {
    other.callback_.f = nullptr;
    other.callback_.param = nullptr;
  }

  ScopedOrtCallbackInvoker& operator=(ScopedOrtCallbackInvoker&& other) noexcept {
    if (callback_.f) {
      callback_.f(callback_.param);
    }

    callback_ = other.callback_;
    other.callback_.f = nullptr;
    other.callback_.param = nullptr;

    return *this;
  }

  ~ScopedOrtCallbackInvoker() noexcept {
    if (callback_.f) {
      callback_.f(callback_.param);
    }
  }

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ScopedOrtCallbackInvoker);
  OrtCallback callback_;
};
}  // namespace onnxruntime
