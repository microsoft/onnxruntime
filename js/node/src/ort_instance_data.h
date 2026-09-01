// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <napi.h>
#include <vector>
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
  static void InitOrt(Napi::Env env, int log_level, Napi::Function tensorConstructor, bool is_main_thread);
  // Get the Tensor constructor reference for the Napi::Env
  static const Napi::FunctionReference& TensorConstructor(Napi::Env env);
  // Return whether this Napi::Env belongs to an Electron runtime.
  static bool IsElectron(Napi::Env env);

  using OutputBufferLease = std::shared_ptr<Napi::ObjectReference>;

  // Acquire a lease for a preallocated output resource in this Napi environment.
  static OutputBufferLease AcquireOutputBufferLease(Napi::Object resource);
  // Release a previously acquired preallocated output resource lease.
  static void ReleaseOutputBufferLease(Napi::Env env, const OutputBufferLease& lease);

 private:
  OrtInstanceData();

  // per env persistent constructors
  Napi::FunctionReference wrappedSessionConstructor;
  Napi::FunctionReference ortTensorConstructor;
  bool isElectron_{false};
  std::vector<OutputBufferLease> outputBufferLeases;
};
