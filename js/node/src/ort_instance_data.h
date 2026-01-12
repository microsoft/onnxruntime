// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>
#include "onnxruntime_cxx_api.h"

/**
 * The OrtInstanceData class is designed to manage the lifecycle of necessary instance data, including:
 * - The Object reference of the InferenceSessionWrap class and the Tensor constructor.
 *   This is a per-env data that has the same lifecycle as the Napi::Env. If there are worker threads, each thread will
 *   have its own handle to the InferenceSessionWrap class and the Tensor constructor.
 *
 * The OrtInstanceData class is bind to the Napi::Env using environment life cycle APIs.
 * see https://nodejs.org/api/n-api.html#environment-life-cycle-apis
 */
struct OrtInstanceData {
  // Create a new OrtInstanceData object related to the Napi::Env
  static void Create(Napi::Env env, Napi::Function inferenceSessionWrapperFunction);
  // Initialize Ort for the Napi::Env
  static void InitOrt(Napi::Env env, int log_level, Napi::Function tensorConstructor);
  // Get the Tensor constructor reference for the Napi::Env
  static const Napi::FunctionReference& TensorConstructor(Napi::Env env);

 private:
  OrtInstanceData();

  // per env persistent constructors
  Napi::FunctionReference wrappedSessionConstructor;
  Napi::FunctionReference ortTensorConstructor;
};
