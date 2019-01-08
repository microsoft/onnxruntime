// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _WIN32
#include <mutex>
namespace onnxruntime {
  using OrtMutex = mutex;
}
#else
#ifdef USE_NSYNC
#include "nsync.h"
#else
#include "pthread.h"
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

 private:
  OrtMutex(const OrtMutex&);             // = delete;
  OrtMutex& operator=(const OrtMutex&);  // = delete;

 public:
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
};  // namespace onnxruntime
#endif