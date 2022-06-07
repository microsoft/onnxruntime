// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <string>
#include <unordered_map>

#include "core/platform/ort_mutex.h"

#ifdef ENABLE_NVTX_PROFILE

namespace onnxruntime {
namespace profile {

// Singleton class of managing global NVTX profiling information.
class Context {
 public:
  static Context& GetInstance() {
    static Context instance_;
    return instance_;
  }

  // Return tag for the specified thread.
  // If the thread's tag doesn't exist, this function returns an empty string.
  std::string GetThreadTagOrDefault(const std::thread::id& thread_id) {
    const std::lock_guard<OrtMutex> lock(mtx_);
    return thread_tag_[thread_id];
  }

  // Set tag for the specified thread.
  void SetThreadTag(
      const std::thread::id& thread_id, const std::string& tag) {
    const std::lock_guard<OrtMutex> lock(mtx_);
    thread_tag_[thread_id] = tag;
  }

 private:
  Context() = default;
  ~Context() = default;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // map from thread's id to its human-readable tag.
  std::unordered_map<std::thread::id, std::string> thread_tag_;
  OrtMutex mtx_;
};

}  // namespace profile
}  // namespace onnxruntime

#endif
