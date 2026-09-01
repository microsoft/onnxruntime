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

  // A region of a preallocated output resource that a run is going to write. 'byteLength' of 0
  // means the whole resource, which is how non-ArrayBuffer resources (a gpu-buffer External) are
  // leased since they have no addressable sub-range.
  struct OutputBufferRegion {
    Napi::ObjectReference resource;
    size_t byteOffset{0};
    size_t byteLength{0};
  };
  using OutputBufferLease = std::shared_ptr<OutputBufferRegion>;

  // Acquire a lease for the region of a preallocated output resource that a run will write.
  // Overlapping regions of the same resource cannot be leased twice at once; disjoint regions can.
  static OutputBufferLease AcquireOutputBufferLease(Napi::Object resource, size_t byteOffset, size_t byteLength);
  // Release a previously acquired preallocated output resource lease.
  static void ReleaseOutputBufferLease(Napi::Env env, const OutputBufferLease& lease);

 private:
  OrtInstanceData();

  // per env persistent constructors
  Napi::FunctionReference wrappedSessionConstructor;
  Napi::FunctionReference ortTensorConstructor;
  std::vector<OutputBufferLease> outputBufferLeases;
};
