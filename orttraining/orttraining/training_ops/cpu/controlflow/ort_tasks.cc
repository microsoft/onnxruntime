// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_tasks.h"

namespace onnxruntime {
namespace contrib {

void OrtTasks::PrepareForegroundWait() {
  std::unique_lock<std::mutex> lock(fg_task.mutex);
  fg_task.signaled.store(false);
}

void OrtTasks::WaitInForegroundThread() {
  std::unique_lock<std::mutex> lock(fg_task.mutex);
  fg_task.cv.wait(lock, [this] { return fg_task.signaled.load(); });
  fg_task.signaled.store(false);
}

void OrtTasks::WakeupForegroundThread() {
  std::unique_lock<std::mutex> lock(fg_task.mutex);
  fg_task.signaled.store(true);
  lock.unlock();
  fg_task.cv.notify_all();
}

void OrtTasks::CreateBackgroundTask(bool* terminate_flags) {
  int64_t run_id = hasher_(std::this_thread::get_id());
  bg_tasks[run_id] = std::make_unique<Task>();
  bg_tasks[run_id]->terminate_flags_ = terminate_flags;
}

void OrtTasks::PrepareBackgroundWait() {
  int64_t run_id = hasher_(std::this_thread::get_id());

  std::unique_lock<std::mutex> lock(bg_tasks[run_id]->mutex);
  bg_tasks[run_id]->signaled.store(false);
}

void OrtTasks::WaitInBackgroundThread() {
  int64_t run_id = hasher_(std::this_thread::get_id());

  std::unique_lock<std::mutex> lock(bg_tasks[run_id]->mutex);
  bg_tasks[run_id]->cv.wait(lock, [this, run_id] { return bg_tasks[run_id]->signaled.load(); });
  bg_tasks[run_id]->signaled.store(false);
}

void OrtTasks::WakeupBackgroundThread(int64_t run_id) {
  std::unique_lock<std::mutex> lock(bg_tasks[run_id]->mutex);
  bg_tasks[run_id]->signaled.store(true);
  lock.unlock();
  bg_tasks[run_id]->cv.notify_all();
}

void OrtTasks::SetForwardOutputs(const std::vector<OrtValue>& forward_outputs) {
  int64_t run_id = hasher_(std::this_thread::get_id());
  bg_tasks[run_id]->forward_output_promise_.set_value(forward_outputs);
}

std::vector<OrtValue> OrtTasks::GetForwardOutputs(int64_t run_id) {
  return bg_tasks[run_id]->forward_output_future_.get();
}

void OrtTasks::SetBackwardInputs(int64_t run_id, const std::vector<OrtValue>& backward_inputs) {
  bg_tasks[run_id]->backward_input_promise_.set_value(backward_inputs);
}

std::vector<OrtValue> OrtTasks::GetBackwardInputs() {
  int64_t run_id = hasher_(std::this_thread::get_id());
  return bg_tasks[run_id]->backward_input_future_.get();
}

void OrtTasks::SetStatus(const Status& status) {
  int64_t run_id = hasher_(std::this_thread::get_id());
  bg_tasks[run_id]->status_promise_.set_value(status);
}

bool OrtTasks::StatusIsReady(int64_t run_id) {
  return bg_tasks[run_id]->status_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

bool OrtTasks::StatusIsValid(int64_t run_id) {
  return bg_tasks[run_id]->status_future_.valid();
}

Status OrtTasks::GetStatus(int64_t run_id) {
  return bg_tasks[run_id]->status_future_.get();
}

}  // namespace contrib
}  // namespace onnxruntime
