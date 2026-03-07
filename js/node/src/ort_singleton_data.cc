// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <mutex>

#include "ort_singleton_data.h"

namespace {
std::mutex ort_singleton_mutex;
OrtSingletonData::OrtObjects* ort_objects = nullptr;
std::atomic<int> ref_count{0};
}  // namespace

OrtSingletonData::OrtObjects::OrtObjects(int log_level)
    : env{OrtLoggingLevel(log_level), "onnxruntime-node"},
      default_run_options{} {
}

OrtSingletonData::OrtObjects& OrtSingletonData::GetOrCreateOrtObjects(napi_env env, int log_level,
                                                                      bool is_main_thread) {
  {
    std::lock_guard<std::mutex> lock(ort_singleton_mutex);
    if (!ort_objects) {
      ort_objects = new OrtObjects(log_level);
    }
    ref_count++;
  }

  // Register a cleanup hook for this napi_env. The hook will be called when this env is torn down.
  // We encode the is_main_thread flag directly into the void* arg to avoid a heap allocation.
  napi_add_env_cleanup_hook(env, CleanupHook, reinterpret_cast<void*>(static_cast<uintptr_t>(is_main_thread)));

  return *ort_objects;
}

void OrtSingletonData::CleanupHook(void* arg) {
  bool is_main_thread = static_cast<bool>(reinterpret_cast<uintptr_t>(arg));

  std::lock_guard<std::mutex> lock(ort_singleton_mutex);
  ref_count--;

  if (ref_count == 0 && is_main_thread) {
    delete ort_objects;
    ort_objects = nullptr;
  }
}

const Ort::Env& OrtSingletonData::Env() {
  return ort_objects->env;
}

const Ort::RunOptions& OrtSingletonData::DefaultRunOptions() {
  return ort_objects->default_run_options;
}
