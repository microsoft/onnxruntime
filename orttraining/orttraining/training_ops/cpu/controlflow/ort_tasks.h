// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/ml_value.h"

#include <mutex>
#include <future>

namespace onnxruntime {
namespace contrib {

struct ForwardReturnType {
  Status status;
  int32_t token_id;
  std::vector<OrtValue> values;
};

struct BackwardReturnType {
  bool terminate;
  int32_t token_id;
  std::vector<OrtValue> values;
};

// TOken IDs are added to events to keep trace of the different
// reasons for passing control between ORT and Python.  They should
// probably just be a debugging aid, with shared knowledge of the
// model being used to keep the pieces in sync.  Currently the
// proof-of-concept uses the values to dispatch to different code
// paths.
//
// These numbers must match ortmodule.py.  In addition,
// ORTModule.max_id must be <= the difference between successive
// values here.  For instance, if we assign IDs [0,100) to custom
// autograd functions then we need a range [100,200) for the tokens
// returned between TOKEN_HOLE_FORWARD and TOKEN_YIELD_END_FORWARD.
static constexpr int TOKEN_HOLE_FORWARD = 100;
static constexpr int TOKEN_YIELD_END_FORWARD = 200;
static constexpr int TOKEN_HOLE_BACKWARD = 300;
static constexpr int TOKEN_END_BACKWARD = 400;

class OrtTasks final {
 public:
  static OrtTasks& GetInstance() {
    static OrtTasks* instance_ = new OrtTasks;
    return *instance_;
  }

  void CreateBackgroundTask(int64_t run_id);
  void RemoveTask(int64_t run_id);

  void SetKernelOutputs(Status s, int32_t token_id, const std::vector<OrtValue>& forward_outputs);
  ForwardReturnType WaitForKernelOutputs(int64_t run_id);
  bool KernelOutputsIsValid();

  void SetExternalKernelOutputs(int64_t run_id, int32_t token_id, const std::vector<OrtValue>& backward_inputs, bool terminate);
  BackwardReturnType WaitForExternalKernelOutputs();

  void SetStatus(const Status& status);
  Status WaitForStatus(int64_t run_id);
  bool TaskIsCompleted(int64_t run_id);

  void CompleteTask(Status s, int32_t token_id);

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
