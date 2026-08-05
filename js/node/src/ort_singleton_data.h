// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>
#include "onnxruntime_cxx_api.h"

/**
 * The OrtSingletonData class is designed to manage the lifecycle of necessary singleton instance data, including:
 *
 * - The Ort::Env singleton instance.
 *   This is a global singleton that is shared across all InferenceSessionWrap instances. It is created when the first
 *   time `InferenceSession.initOrtOnce()` is called.
 *
 * - The Ort::RunOptions singleton instance.
 *   This is an empty default RunOptions instance. It is created once to allow reuse across all session inference runs.
 *
 * The OrtSingletonData class uses a ref-counted, heap-allocated singleton with best-effort cleanup.
 *
 * Each napi_env (one per thread) that initializes ORT increments a ref count and registers a cleanup hook via
 * napi_add_env_cleanup_hook. When the hook fires, the ref count is decremented. The singleton is only deleted when:
 *   1. The ref count reaches 0, AND
 *   2. The cleanup hook is running on the main thread (determined by the isMainThread flag from worker_threads).
 *
 * This ensures:
 * - On normal single-threaded usage, the singleton is properly destroyed (no leak).
 * - On multi-threaded usage where workers exit before the main thread, the main thread's hook fires last with
 *   ref count 0 and performs the cleanup.
 * - If cleanup hooks don't fire (e.g., uncaught exception — see https://github.com/nodejs/node/issues/58341),
 *   the ref count stays >0 and the singleton safely leaks, avoiding crashes from calling into an already-unloaded
 *   onnxruntime shared library.
 * - If the main thread's hook fires but workers are still alive (e.g., process.exit()), the ref count is >0 and
 *   the singleton safely leaks.
 */
struct OrtSingletonData {
  struct OrtObjects {
    Ort::Env env;
    Ort::RunOptions default_run_options;

   private:
    // The following pattern ensures that OrtObjects can only be created by OrtSingletonData
    OrtObjects(int log_level);
    friend struct OrtSingletonData;
  };

  // Initialize ORT objects and register a cleanup hook for the given napi_env.
  // Each napi_env (thread) should call this once.
  // is_main_thread should be set to true if the calling thread is the main thread (from worker_threads.isMainThread).
  static void InitOrtObjects(napi_env env, int log_level, bool is_main_thread);

  // Get the ORT singleton objects. Returns nullptr if the singleton has been destroyed.
  static OrtObjects* GetOrtObjects();

 private:
  static void CleanupHook(void* arg);
};
