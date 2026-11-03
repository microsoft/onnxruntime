// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <mutex>

/**
 * Process-wide serialization of work on execution providers with shared device state.
 *
 * ORT's per-session guard for providers that declare ConcurrentRunSupported() false covers graph execution only.
 * WebGPU sessions on the same device also share one WebGpuContext, and with it one command encoder and buffer
 * manager, so concurrent use from two sessions corrupts it. Everything that touches that state is therefore done
 * while holding this lock: creating and destroying such a session, binding, running, and releasing device-backed
 * OrtValues. Sessions on other providers never take it.
 *
 * The JavaScript thread must not block on the lock, since its holder may be in the middle of a long inference.
 * Releases are posted with ReleaseUnderDeviceLock() instead, and run right away when the lock is free or by the
 * holder before it unlocks.
 */
class DeviceLock {
 public:
  // Acquires the lock if `required`; otherwise holds nothing.
  explicit DeviceLock(bool required);
  ~DeviceLock();

  DeviceLock(const DeviceLock&) = delete;
  DeviceLock& operator=(const DeviceLock&) = delete;
  DeviceLock(DeviceLock&&) = delete;
  DeviceLock& operator=(DeviceLock&&) = delete;

 private:
  // Recursive because a posted release can drop the last reference to another device object, which posts again
  // from the thread that is already holding the lock.
  std::unique_lock<std::recursive_mutex> lock_;
};

// Runs `release` under the device lock without blocking the calling thread.
void ReleaseUnderDeviceLock(std::function<void()> release);
