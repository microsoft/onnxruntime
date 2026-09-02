// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
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

  // A region of a preallocated output resource that a run is going to write.
  struct OutputBufferRegion {
    Napi::ObjectReference resource;
    size_t byteOffset{0};
    size_t byteLength{0};
    // The lease covers the whole resource. Used for a gpu-buffer External, which has no addressable
    // sub-range; distinct from a zero-length region, which writes nothing and conflicts with nothing.
    bool wholeResource{false};
  };
  // Serializes every operation that touches execution provider device state: binding inputs and
  // outputs, running with an IoBinding, and releasing device-backed OrtValues. Providers that
  // declare ConcurrentRunSupported() false are not safe for concurrent use, and ORT's own guard
  // covers only graph execution; WebGPU sessions sharing a device id also share one WebGpuContext
  // and command encoder, so the lock has to span sessions rather than sit on one of them.
  static std::mutex& DeviceMutex();

  // Destroy a device-backed object under DeviceMutex(). Safe to call from the Javascript thread:
  // blocking there would stall the event loop for the length of another session's inference, so if
  // the lock is busy the object is queued and destroyed by whoever holds the lock next.
  static void ReleaseDeviceObject(std::shared_ptr<void> object);
  // Destroy everything queued by ReleaseDeviceObject(). The caller must already hold DeviceMutex().
  static void DrainDeviceReleasesLocked();

  // Whether a previous attempt to hand Javascript an external ArrayBuffer was refused. Electron's
  // V8 Memory Cage and V8-sandbox builds reject them for the lifetime of the process, so the answer
  // is cached rather than re-probed for every model output.
  static bool ExternalArrayBuffersRefused(Napi::Env env);
  static void MarkExternalArrayBuffersRefused(Napi::Env env);

  using OutputBufferLease = std::shared_ptr<OutputBufferRegion>;

  // Acquire a lease for the region of a preallocated output resource that a run will write.
  // Overlapping regions of the same resource cannot be leased twice at once; disjoint regions can.
  static OutputBufferLease AcquireOutputBufferLease(Napi::Object resource, size_t byteOffset, size_t byteLength,
                                                    bool wholeResource);
  // Release a previously acquired preallocated output resource lease.
  static void ReleaseOutputBufferLease(Napi::Env env, const OutputBufferLease& lease);

 private:
  OrtInstanceData();

  // per env persistent constructors
  Napi::FunctionReference wrappedSessionConstructor;
  Napi::FunctionReference ortTensorConstructor;
  std::vector<OutputBufferLease> outputBufferLeases;
  bool externalArrayBuffersRefused{false};
};
