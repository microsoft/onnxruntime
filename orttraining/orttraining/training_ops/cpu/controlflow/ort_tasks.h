// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <memory>
#include <thread>
#include <future>
#include <condition_variable>
#include "core/framework/ml_value.h"

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

  void CreateBackgroundTask(std::promise<std::vector<OrtValue>> forward_output_promise,
                            std::promise<std::vector<OrtValue>> backward_input_promise);
  void PrepareBackgroundWait();
  void WaitInBackgroundThread();
  void WakeupBackgroundThread(int64_t run_id);

  void SetForwardOutputs(const std::vector<OrtValue>& forward_outputs);
  std::vector<OrtValue> GetForwardOutputs(int64_t run_id);

  void SetBackwardInputs(int64_t run_id, const std::vector<OrtValue>& backward_inputs);
  std::vector<OrtValue> GetBackwardInputs();

 private:
  OrtTasks() = default;
  ~OrtTasks() = default;
  OrtTasks(const OrtTasks&) = delete;
  OrtTasks& operator=(const OrtTasks&) = delete;

  struct Item {
    std::atomic<bool> signaled;
    mutable std::mutex mutex;
    mutable std::condition_variable cv;

    std::promise<std::vector<OrtValue>> forward_output_promise_;
    std::promise<std::vector<OrtValue>> backward_input_promise_;

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
