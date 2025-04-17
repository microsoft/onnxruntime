// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>
#include "onnxruntime_cxx_api.h"

/**
 * The OrtInstanceData class is designed to manage the lifecycle of necessary instance data, including:
 * - The Ort::Env singleton instance.
 *   This is a global singleton that is shared across all InferenceSessionWrap instances. It is created when the first
 *   time `InferenceSession.initOrtOnce()` is called. It is destroyed when the last active NAPI Env is destroyed.
 *   Once destroyed, it cannot be created again.
 *
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
  // Get the global Ort::Env
  static const Ort::Env* OrtEnv() { return ortEnv.get(); }
  // Get the default Ort::RunOptions
  static Ort::RunOptions* OrtDefaultRunOptions() { return ortDefaultRunOptions.get(); }

  ~OrtInstanceData();

 private:
  OrtInstanceData();

  // per env persistent constructors
  Napi::FunctionReference wrappedSessionConstructor;
  Napi::FunctionReference ortTensorConstructor;

  // ORT env (global singleton)
  static std::unique_ptr<Ort::Env> ortEnv;
  static std::unique_ptr<Ort::RunOptions> ortDefaultRunOptions;
  static std::mutex ortEnvMutex;
  static std::atomic<uint64_t> ortEnvRefCount;
  static std::atomic<bool> ortEnvDestroyed;
};
