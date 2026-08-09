// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <mutex>

#include "ort_singleton_data.h"

namespace {
std::mutex ort_singleton_mutex;
std::atomic<OrtSingletonData::OrtObjects*> ort_objects{nullptr};
std::atomic<int> ref_count{0};
}  // namespace

OrtSingletonData::OrtObjects::OrtObjects(int log_level)
    : env{OrtLoggingLevel(log_level), "onnxruntime-node"},
      default_run_options{} {
}

void OrtSingletonData::InitOrtObjects(napi_env env, int log_level,
                                      bool is_main_thread) {
  {
    std::lock_guard<std::mutex> lock(ort_singleton_mutex);
    if (!ort_objects.load(std::memory_order_relaxed)) {
      ort_objects.store(new OrtObjects(log_level), std::memory_order_release);
    }
    ref_count++;
  }

  // Register a cleanup hook for this napi_env. The hook will be called when this env is torn down.
  // We encode the is_main_thread flag directly into the void* arg to avoid a heap allocation.
  napi_add_env_cleanup_hook(env, CleanupHook, reinterpret_cast<void*>(static_cast<uintptr_t>(is_main_thread)));
}

void OrtSingletonData::CleanupHook(void* arg) {
  bool is_main_thread = static_cast<bool>(reinterpret_cast<uintptr_t>(arg));

  std::lock_guard<std::mutex> lock(ort_singleton_mutex);
  ref_count--;

  if (ref_count == 0 && is_main_thread) {
    delete ort_objects.load(std::memory_order_relaxed);
    ort_objects.store(nullptr, std::memory_order_release);
  }
}

OrtSingletonData::OrtObjects* OrtSingletonData::GetOrtObjects() {
  return ort_objects.load(std::memory_order_acquire);
}
