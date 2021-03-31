// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/ml_value.h"

#include <mutex>
#include <future>


namespace onnxruntime {
namespace contrib {

// the pair is <forward_status, forward_outputs>
typedef std::pair<Status, std::vector<OrtValue>> ForwardReturnType;

// the pair is <terminate_flag, backward_inputs>
typedef std::pair<bool, std::vector<OrtValue>> BackwardReturnType;

class OrtTasks final {
 public:
  static OrtTasks& GetInstance() {
    static OrtTasks instance_;
    return instance_;
  }

  void CreateBackgroundTask(int64_t run_id);
  void RemoveTask(int64_t run_id);

  void SetForwardOutputs(Status s, const std::vector<OrtValue>& forward_outputs);
  ForwardReturnType WaitForForwardOutputs(int64_t run_id);
  bool ForwardOutputsIsValid() const;

  void SetBackwardInputs(int64_t run_id, const std::vector<OrtValue>& backward_inputs, bool terminate);
  BackwardReturnType WaitForBackwardInputs();

  void SetStatus(const Status& status);
  Status WaitForStatus(int64_t run_id);
  bool TaskIsCompleted(int64_t run_id) const;

 private:
  OrtTasks() = default;
  ~OrtTasks() = default;
  OrtTasks(const OrtTasks&) = delete;
  OrtTasks& operator=(const OrtTasks&) = delete;

  struct Task {
    std::promise<ForwardReturnType> forward_output_promise_;
    std::future<ForwardReturnType> forward_output_future_ = forward_output_promise_.get_future();

    std::promise<BackwardReturnType> backward_input_promise_;
    std::future<BackwardReturnType> backward_input_future_ = backward_input_promise_.get_future();

    std::promise<Status> status_promise_;
    std::future<Status> status_future_ = status_promise_.get_future();
  };

  std::hash<std::thread::id> hasher_;
  mutable std::mutex mutex_;
  std::unordered_map<int64_t, std::unique_ptr<Task>> bg_tasks_;
};

}  // namespace contrib
}  // namespace onnxruntime
