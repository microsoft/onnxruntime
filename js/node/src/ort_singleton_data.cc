// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_singleton_data.h"

OrtSingletonData::OrtObjects::OrtObjects(int log_level)
    : env(new Ort::Env{OrtLoggingLevel(log_level), "onnxruntime-node"}),
      default_run_options(new Ort::RunOptions{}) {
}

OrtSingletonData::OrtObjects& OrtSingletonData::GetOrCreateOrtObjects(int log_level) {
  static OrtObjects ort_objects(log_level);
  return ort_objects;
}

const Ort::Env* OrtSingletonData::Env() {
  return GetOrCreateOrtObjects().env.get();
}

const Ort::RunOptions* OrtSingletonData::DefaultRunOptions() {
  return GetOrCreateOrtObjects().default_run_options.get();
}
