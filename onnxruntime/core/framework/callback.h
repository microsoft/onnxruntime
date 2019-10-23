// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

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
  OrtCallback callback{nullptr, nullptr};

  template <typename T>
  void operator()(T) {
    if (callback.f) {
      callback.f(callback.param);
    }
  }
};
}  // namespace onnxruntime