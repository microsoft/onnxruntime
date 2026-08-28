// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>

#include "onnxruntime_cxx_api.h"

/**
 * EpContextDataReadState owns the JavaScript callback registered through
 * `InferenceSession.SessionOptions.epContextDataRead` and adapts it to OrtReadNamedBufferFunc.
 *
 * Threading model:
 * - The state is created on the JavaScript thread that parses the session options, and it keeps a
 *   thread-safe function that is used to reach that thread again.
 * - ONNX Runtime may invoke the read callback from any of its worker threads and does not serialize
 *   calls, so every invocation is serialized here before JavaScript runs.
 * - No exception ever crosses the C ABI: every failure is converted to an OrtStatus.
 *
 * Lifetime model:
 * - The state is reference counted. The InferenceSession wrapper holds a strong reference for as long
 *   as its native session exists, and the asynchronous session construction takes its own strong
 *   snapshot, so the state always outlives any ONNX Runtime code that can call it.
 * - `Release()` gives up the thread-safe function. It must be called on the JavaScript thread, and only
 *   once the native session that could invoke the callback has been released.
 * - After environment teardown the N-API references are dropped by the thread-safe function finalizer,
 *   and later calls fail with a status instead of touching N-API.
 */
class EpContextDataReadState final {
 public:
  /**
   * Create the state on the JavaScript thread.
   * @throw Napi::Error if the thread-safe function cannot be created.
   */
  static std::shared_ptr<EpContextDataReadState> Create(Napi::Env env, Napi::Function callback, size_t maxDataSize);

  ~EpContextDataReadState();

  EpContextDataReadState(const EpContextDataReadState&) = delete;
  EpContextDataReadState& operator=(const EpContextDataReadState&) = delete;
  EpContextDataReadState(EpContextDataReadState&&) = delete;
  EpContextDataReadState& operator=(EpContextDataReadState&&) = delete;

  /**
   * OrtReadNamedBufferFunc implementation. `state` must be an EpContextDataReadState*.
   */
  static OrtStatus* ORT_API_CALL ReadNamedBuffer(void* state, const char* name, OrtAllocator* allocator,
                                                 void** buffer, size_t* data_size) noexcept;

  /**
   * Give up the thread-safe function. Idempotent, and safe to call during environment teardown.
   * Must be called on the JavaScript thread that created the state, and only after the native session
   * that could trigger callbacks has been released.
   */
  void Release() noexcept;

  size_t MaxDataSize() const noexcept { return max_data_size_; }

 private:
  struct CallItem;

  explicit EpContextDataReadState(size_t maxDataSize);

  OrtStatus* Invoke(const char* name, OrtAllocator* allocator, void** buffer, size_t* data_size) noexcept;
  void InvokeOnOwnerThread(CallItem& item) noexcept;
  void OnThreadSafeFunctionFinalize() noexcept;

  static void RunCallback(napi_env env, napi_value jsCallback, CallItem& item) noexcept;
  static void CallJs(napi_env env, napi_value jsCallback, void* context, void* data);
  static void FinalizeThreadSafeFunction(napi_env env, void* finalizeData, void* hint);

  const size_t max_data_size_;
  const std::thread::id owner_thread_id_;

  // Only touched on the JavaScript thread.
  Napi::FunctionReference callback_ref_;

  std::atomic<napi_threadsafe_function> tsfn_{nullptr};
  std::atomic<bool> tsfn_released_{false};
  std::atomic<bool> js_available_{false};

  // Serializes callback invocations coming from different ONNX Runtime threads.
  std::mutex call_mutex_;
};

/**
 * Parse `sessionOptions.epContextDataRead` and register the callback on the native session options.
 *
 * The setup is transactional: the new state is created and retained first, the native setter is invoked
 * next, and `state` is only replaced after the setter succeeded. A previously published state is
 * released when it is replaced.
 *
 * @throw Napi::TypeError / Napi::RangeError for invalid configuration.
 */
void ParseEpContextDataReadOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions,
                                   std::shared_ptr<EpContextDataReadState>& state);
