// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_tasks.h"

namespace onnxruntime {
namespace contrib {

void OrtTasks::CreateBackgroundTask(int64_t run_id, bool* terminate_flags) {
  bg_tasks_[run_id] = std::make_unique<Task>();
  bg_tasks_[run_id]->terminate_flags_ = terminate_flags;
}

void OrtTasks::SetForwardOutputs(const std::vector<OrtValue>& forward_outputs) {
  int64_t run_id = hasher_(std::this_thread::get_id());
  bg_tasks_[run_id]->forward_output_promise_.set_value(forward_outputs);
}

std::vector<OrtValue> OrtTasks::WaitForForwardOutputs(int64_t run_id) {
  return bg_tasks_[run_id]->forward_output_future_.get();
}

bool OrtTasks::ForwardOutputsIsValid() {
  int64_t run_id = hasher_(std::this_thread::get_id());
  return bg_tasks_[run_id]->forward_output_future_.valid();
}

void OrtTasks::SetBackwardInputs(int64_t run_id, const std::vector<OrtValue>& backward_inputs) {
  bg_tasks_[run_id]->backward_input_promise_.set_value(backward_inputs);
}

std::vector<OrtValue> OrtTasks::WaitForBackwardInputs() {
  int64_t run_id = hasher_(std::this_thread::get_id());
  return bg_tasks_[run_id]->backward_input_future_.get();
}

void OrtTasks::SetStatus(const Status& status) {
  int64_t run_id = hasher_(std::this_thread::get_id());
  bg_tasks_[run_id]->status_promise_.set_value(status);
}

bool OrtTasks::StatusIsReady(int64_t run_id) {
  return bg_tasks_[run_id]->status_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

bool OrtTasks::StatusIsValid(int64_t run_id) {
  return bg_tasks_[run_id]->status_future_.valid();
}

Status OrtTasks::WaitForStatus(int64_t run_id) {
  return bg_tasks_[run_id]->status_future_.get();
}

}  // namespace contrib
}  // namespace onnxruntime
