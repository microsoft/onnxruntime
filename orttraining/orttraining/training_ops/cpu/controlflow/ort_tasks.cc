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

void OrtTasks::PrepareBackgroundWait() {
  std::unique_lock<std::mutex> lock(bg_event_.mutex);
  bg_event_.signaled.store(false);
}

void OrtTasks::WaitInBackgroundThread() {
  std::unique_lock<std::mutex> lock(bg_event_.mutex);
  bg_event_.cv.wait(lock, [this] { return bg_event_.signaled.load(); });
  bg_event_.signaled.store(false);
}

void OrtTasks::WakeupBackgroundThread() {
  std::unique_lock<std::mutex> lock(bg_event_.mutex);
  bg_event_.signaled.store(true);
  lock.unlock();
  bg_event_.cv.notify_all();
}

}  // namespace contrib
}  // namespace onnxruntime
