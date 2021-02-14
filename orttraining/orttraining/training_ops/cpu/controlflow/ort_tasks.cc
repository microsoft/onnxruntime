// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_tasks.h"

namespace onnxruntime {
namespace contrib {

void OrtTasks::PrepareForegroundWait() {
  std::unique_lock<std::mutex> lock(fg_event_.mutex);
  fg_event_.signaled.store(false);
}

void OrtTasks::WaitInForegroundThread() {
  std::unique_lock<std::mutex> lock(fg_event_.mutex);
  fg_event_.cv.wait(lock, [this] { return fg_event_.signaled.load(); });
  fg_event_.signaled.store(false);
}

void OrtTasks::WakeupForegroundThread() {
  std::unique_lock<std::mutex> lock(fg_event_.mutex);
  fg_event_.signaled.store(true);
  lock.unlock();
  fg_event_.cv.notify_all();
}

void OrtTasks::CreateBackgroundTask() {
  int64_t run_id = hasher_(std::this_thread::get_id());

  bg_events_.emplace(run_id, std::make_unique<Item>());
}

void OrtTasks::PrepareBackgroundWait() {
  int64_t run_id = hasher_(std::this_thread::get_id());

  std::unique_lock<std::mutex> lock(bg_events_[run_id]->mutex);
  bg_events_[run_id]->signaled.store(false);
}

void OrtTasks::WaitInBackgroundThread() {
  int64_t run_id = hasher_(std::this_thread::get_id());

  std::unique_lock<std::mutex> lock(bg_events_[run_id]->mutex);
  bg_events_[run_id]->cv.wait(lock, [this, run_id] { return bg_events_[run_id]->signaled.load(); });
  bg_events_[run_id]->signaled.store(false);
}

void OrtTasks::WakeupBackgroundThread(int64_t run_id) {
  std::unique_lock<std::mutex> lock(bg_events_[run_id]->mutex);
  bg_events_[run_id]->signaled.store(true);
  lock.unlock();
  bg_events_[run_id]->cv.notify_all();
}

void OrtTasks::Push(int64_t run_id, const OrtValue& ort_value) {
  bg_events_[run_id]->message_queue_.Push(ort_value);
}

OrtValue OrtTasks::Pop(int64_t run_id) {
  return bg_events_[run_id]->message_queue_.Pop();
}

void OrtTasks::PopAll(int64_t run_id, std::vector<OrtValue>& results) {
  bg_events_[run_id]->message_queue_.PopAll(results);
}

}  // namespace contrib
}  // namespace onnxruntime
