// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_lock.h"

#include <utility>
#include <vector>

namespace {
std::recursive_mutex device_mutex;
std::mutex pending_mutex;
std::vector<std::function<void()>> pending_releases;

bool HasPendingReleases() {
  std::lock_guard<std::mutex> lock(pending_mutex);
  return !pending_releases.empty();
}

// The caller holds device_mutex.
void RunPendingReleases() {
  for (;;) {
    std::vector<std::function<void()>> releases;
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      releases.swap(pending_releases);
    }
    if (releases.empty()) {
      return;
    }
    for (auto& release : releases) {
      release();
    }
  }
}

void TryRunPendingReleases() {
  // A release posted while another thread holds the lock is left to that thread, which drains again after
  // unlocking. The same applies here: anything posted while this thread held the lock is picked up by the loop.
  do {
    std::unique_lock<std::recursive_mutex> lock(device_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      return;
    }
    RunPendingReleases();
  } while (HasPendingReleases());
}
}  // namespace

DeviceLock::DeviceLock(bool required) {
  if (required) {
    lock_ = std::unique_lock<std::recursive_mutex>(device_mutex);
  }
}

DeviceLock::~DeviceLock() {
  if (!lock_.owns_lock()) {
    return;
  }
  RunPendingReleases();
  lock_.unlock();
  // Covers a release posted after the drain above but before the unlock, whose own attempt found the lock taken.
  TryRunPendingReleases();
}

void ReleaseUnderDeviceLock(std::function<void()> release) {
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    pending_releases.push_back(std::move(release));
  }
  TryRunPendingReleases();
}
