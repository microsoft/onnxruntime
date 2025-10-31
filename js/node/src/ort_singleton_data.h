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
 * The OrtSingletonData class uses the "Meyers Singleton" pattern to ensure thread-safe lazy initialization, as well as
 * proper destruction order at program exit.
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

  static OrtObjects& GetOrCreateOrtObjects(int log_level = ORT_LOGGING_LEVEL_WARNING);

  // Get the global Ort::Env
  static const Ort::Env& Env();

  // Get the default Ort::RunOptions
  static const Ort::RunOptions& DefaultRunOptions();
};
