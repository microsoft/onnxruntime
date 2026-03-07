// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_singleton_data.h"

OrtSingletonData::OrtObjects::OrtObjects(int log_level)
    : env{OrtLoggingLevel(log_level), "onnxruntime-node"},
      default_run_options{} {
}

OrtSingletonData::OrtObjects& OrtSingletonData::GetOrCreateOrtObjects(int log_level) {
  // Intentionally leaked to avoid calling destructors at program exit.
  // The destructor of Ort::Env calls OrtApi::ReleaseEnv through a function pointer table that may point into an
  // already-unloaded onnxruntime shared library, causing a crash.
  static OrtObjects* ort_objects = new OrtObjects(log_level);
  return *ort_objects;
}

const Ort::Env& OrtSingletonData::Env() {
  return GetOrCreateOrtObjects().env;
}

const Ort::RunOptions& OrtSingletonData::DefaultRunOptions() {
  return GetOrCreateOrtObjects().default_run_options;
}
