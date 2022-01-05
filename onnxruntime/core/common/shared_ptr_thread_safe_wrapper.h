// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "core/platform/ort_mutex.h"

namespace onnxruntime {

// provides limited thread-safe access to a shared_ptr<T>
// the underlying shared_ptr<T> can, in a thread-safe manner:
// - be copied, holding a value which is initialized on demand, with GetInitialized()
// - be reset with Reset()
// `init_fn` is called to obtain the initialized value
template <typename T>
class SharedPtrThreadSafeWrapper {
 public:
  using InitFn = std::function<std::shared_ptr<T>()>;

  explicit SharedPtrThreadSafeWrapper(InitFn init_fn) : init_fn_{init_fn} {}

  std::shared_ptr<T> GetInitialized() {
    std::scoped_lock lock{ptr_mutex_};
    if (!ptr_) {
      ptr_ = init_fn_();
    }
    return ptr_;
  }

  void Reset() {
    std::scoped_lock lock{ptr_mutex_};
    ptr_.reset();
  }

 private:
  InitFn init_fn_;

  OrtMutex ptr_mutex_;
  std::shared_ptr<T> ptr_;
};

}  // namespace onnxruntime
