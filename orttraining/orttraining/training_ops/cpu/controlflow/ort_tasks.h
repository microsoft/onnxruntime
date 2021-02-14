// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <thread>
#include <condition_variable>

namespace onnxruntime {
namespace contrib {

class OrtTasks final {
 public:
  static OrtTasks& GetInstance() {
    static OrtTasks instance_;
    return instance_;
  }

  void PrepareForegroundWait();
  void WaitInForegroundThread();
  void WakeupForegroundThread();

  void CreateBackgroundTask();
  void PrepareBackgroundWait();
  void WaitInBackgroundThread();
  void WakeupBackgroundThread(int64_t run_id);

 private:
   OrtTasks() = default;
  ~OrtTasks() = default;
  OrtTasks(const OrtTasks&) = delete;
  OrtTasks& operator=(const OrtTasks&) = delete;
  
  struct Item {
    std::atomic<bool> signaled;
    mutable std::mutex mutex;
    mutable std::condition_variable cv;

    Item() {
      signaled.store(false);
    }
  };

  std::hash<std::thread::id> hasher_;
  Item fg_event_; 
  std::unordered_map<int64_t, std::unique_ptr<Item>> bg_events_;
};

}  // namespace contrib
}  // namespace onnxruntime
