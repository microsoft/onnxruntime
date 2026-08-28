// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <napi.h>

#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "common.h"
#include "ep_context_data_read_helper.h"

namespace {

// Number.MAX_SAFE_INTEGER. A larger value cannot round-trip through a JavaScript number.
constexpr double kMaxSafeInteger = 9007199254740991.0;

// RAII owner for a buffer allocated through the OrtAllocator that ONNX Runtime supplies to the read
// callback. The bytes are only detached (handed over to ONNX Runtime) once the whole callback succeeded.
struct AllocatorBuffer {
  OrtAllocator* allocator = nullptr;
  void* data = nullptr;
  size_t size = 0;

  AllocatorBuffer() = default;
  AllocatorBuffer(const AllocatorBuffer&) = delete;
  AllocatorBuffer& operator=(const AllocatorBuffer&) = delete;
  ~AllocatorBuffer() { Reset(); }

  void Reset() noexcept {
    if (data != nullptr && allocator != nullptr) {
      allocator->Free(allocator, data);
    }
    data = nullptr;
    size = 0;
  }

  void* Detach() noexcept {
    void* detached = data;
    data = nullptr;
    return detached;
  }
};

OrtStatus* MakeOrtStatus(OrtErrorCode code, const std::string& message) noexcept {
  return Ort::GetApi().CreateStatus(code, message.c_str());
}

}  // namespace

// One read request. The requesting ONNX Runtime thread blocks on `Wait()` until the JavaScript thread
// has produced a result or reported a failure.
struct EpContextDataReadState::CallItem {
  // inputs
  const char* name = nullptr;
  OrtAllocator* allocator = nullptr;
  size_t max_data_size = 0;

  // outputs
  AllocatorBuffer buffer;
  bool succeeded = false;
  OrtErrorCode error_code = ORT_FAIL;
  std::string error_message;

  std::mutex mutex;
  std::condition_variable cv;
  bool completed = false;

  void Succeed() noexcept { succeeded = true; }

  void Fail(OrtErrorCode code, std::string message) noexcept {
    buffer.Reset();
    succeeded = false;
    error_code = code;
    error_message = std::move(message);
  }

  void Complete() noexcept {
    {
      std::lock_guard<std::mutex> lock{mutex};
      completed = true;
    }
    cv.notify_all();
  }

  void Wait() noexcept {
    std::unique_lock<std::mutex> lock{mutex};
    cv.wait(lock, [this] { return completed; });
  }
};

EpContextDataReadState::EpContextDataReadState(size_t maxDataSize)
    : max_data_size_{maxDataSize}, owner_thread_id_{std::this_thread::get_id()} {}

EpContextDataReadState::~EpContextDataReadState() {
  // The thread-safe function finalizer already dropped `callback_ref_` on the JavaScript thread, so no
  // N-API reference is touched here. This destructor may run long after environment teardown.
  Release();
}

std::shared_ptr<EpContextDataReadState> EpContextDataReadState::Create(Napi::Env env, Napi::Function callback,
                                                                       size_t maxDataSize) {
  std::shared_ptr<EpContextDataReadState> state{new EpContextDataReadState{maxDataSize}};

  state->callback_ref_ = Napi::Persistent(callback);
  // The reference is always dropped explicitly (finalizer or the failure path below). Suppressing the
  // destructor guarantees that the last shared_ptr release never calls into N-API.
  state->callback_ref_.SuppressDestruct();

  Napi::String resourceName = Napi::String::New(env, "onnxruntime.epContextDataRead");

  // The finalizer data keeps the state alive until N-API is done with the thread-safe function, so the
  // finalizer can never observe a destroyed object, including during environment teardown.
  auto* finalizeData = new std::shared_ptr<EpContextDataReadState>{state};

  napi_threadsafe_function tsfn = nullptr;
  napi_status status = napi_create_threadsafe_function(
      env, callback, /* async_resource */ nullptr, resourceName,
      /* max_queue_size */ 0, /* initial_thread_count */ 1, finalizeData,
      &EpContextDataReadState::FinalizeThreadSafeFunction, state.get(), &EpContextDataReadState::CallJs, &tsfn);
  if (status != napi_ok) {
    delete finalizeData;
    state->callback_ref_.Reset();
    ORT_NAPI_THROW_ERROR(env, "Failed to register 'sessionOptions.epContextDataRead.callback'.");
  }

  state->tsfn_.store(tsfn, std::memory_order_release);
  state->js_available_.store(true, std::memory_order_release);

  // The callback must not keep the Node.js event loop alive on its own: the pending asynchronous session
  // construction already holds the event loop open for as long as callbacks can be delivered.
  napi_unref_threadsafe_function(env, tsfn);

  return state;
}

void EpContextDataReadState::Release() noexcept {
  if (tsfn_released_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  js_available_.store(false, std::memory_order_release);

  napi_threadsafe_function tsfn = tsfn_.exchange(nullptr, std::memory_order_acq_rel);
  if (tsfn != nullptr) {
    napi_release_threadsafe_function(tsfn, napi_tsfn_release);
  }
}

void EpContextDataReadState::OnThreadSafeFunctionFinalize() noexcept {
  // Runs on the JavaScript thread while the environment is still usable, including during teardown.
  js_available_.store(false, std::memory_order_release);
  tsfn_released_.store(true, std::memory_order_release);
  tsfn_.store(nullptr, std::memory_order_release);
  callback_ref_.Reset();
}

void EpContextDataReadState::FinalizeThreadSafeFunction(napi_env /* env */, void* finalizeData, void* /* hint */) {
  auto* holder = static_cast<std::shared_ptr<EpContextDataReadState>*>(finalizeData);
  if (holder == nullptr) {
    return;
  }
  (*holder)->OnThreadSafeFunctionFinalize();
  delete holder;
}

void EpContextDataReadState::RunCallback(napi_env rawEnv, napi_value jsCallback, CallItem& item) noexcept {
  try {
    Napi::Env env{rawEnv};
    Napi::HandleScope scope{env};

    Napi::Function callback{env, jsCallback};
    Napi::Value result = callback.Call({Napi::String::New(env, item.name)});

    // A Node.js Buffer is a Uint8Array, so it is accepted by this check. Any other type is not.
    if (!result.IsTypedArray() || result.As<Napi::TypedArray>().TypedArrayType() != napi_uint8_array) {
      item.Fail(ORT_INVALID_ARGUMENT,
                MakeString("'sessionOptions.epContextDataRead.callback' must return a Uint8Array, but it returned "
                           "another type for '",
                           item.name, "'."));
      return;
    }

    auto typedArray = result.As<Napi::TypedArray>();
    const size_t byteLength = typedArray.ByteLength();
    if (byteLength > item.max_data_size) {
      item.Fail(ORT_INVALID_ARGUMENT,
                MakeString("'sessionOptions.epContextDataRead.callback' returned ", byteLength, " bytes for '",
                           item.name, "', which exceeds 'sessionOptions.epContextDataRead.maxDataSize' (",
                           item.max_data_size, ")."));
      return;
    }

    // An empty payload is reported as {nullptr, 0}; nothing is allocated for it.
    if (byteLength == 0) {
      item.Succeed();
      return;
    }

    void* allocated = item.allocator->Alloc(item.allocator, byteLength);
    if (allocated == nullptr) {
      item.Fail(ORT_FAIL, MakeString("Failed to allocate ", byteLength, " bytes for the EPContext data of '",
                                     item.name, "'."));
      return;
    }
    item.buffer.allocator = item.allocator;
    item.buffer.data = allocated;
    item.buffer.size = byteLength;

    const auto* source =
        reinterpret_cast<const uint8_t*>(typedArray.ArrayBuffer().Data()) + typedArray.ByteOffset();
    std::memcpy(allocated, source, byteLength);
    item.Succeed();
  } catch (Napi::Error const& e) {
    item.Fail(ORT_FAIL, MakeString("'sessionOptions.epContextDataRead.callback' failed for '", item.name,
                                   "': ", e.Message()));
  } catch (std::exception const& e) {
    item.Fail(ORT_FAIL, MakeString("'sessionOptions.epContextDataRead.callback' failed for '", item.name,
                                   "': ", e.what()));
  } catch (...) {
    item.Fail(ORT_FAIL, MakeString("'sessionOptions.epContextDataRead.callback' failed for '", item.name,
                                   "' with an unknown error."));
  }
}

void EpContextDataReadState::CallJs(napi_env env, napi_value jsCallback, void* /* context */, void* data) {
  auto* payload = static_cast<std::shared_ptr<CallItem>*>(data);
  if (payload == nullptr) {
    return;
  }
  std::shared_ptr<CallItem> item = *payload;
  delete payload;

  if (env == nullptr || jsCallback == nullptr) {
    // N-API drains the queue with null arguments when the environment can no longer run JavaScript.
    item->Fail(ORT_FAIL,
               "The JavaScript environment was torn down before 'sessionOptions.epContextDataRead.callback' ran.");
  } else {
    RunCallback(env, jsCallback, *item);
  }
  item->Complete();
}

void EpContextDataReadState::InvokeOnOwnerThread(CallItem& item) noexcept {
  try {
    Napi::Env env = callback_ref_.Env();
    Napi::HandleScope scope{env};
    RunCallback(env, callback_ref_.Value(), item);
  } catch (...) {
    item.Fail(ORT_FAIL, "Failed to enter the JavaScript scope for 'sessionOptions.epContextDataRead.callback'.");
  }
}

OrtStatus* EpContextDataReadState::Invoke(const char* name, OrtAllocator* allocator, void** buffer,
                                          size_t* data_size) noexcept {
  // The outputs are initialized before anything else runs, so ONNX Runtime never observes stale values.
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (data_size != nullptr) {
    *data_size = 0;
  }
  if (buffer == nullptr || data_size == nullptr || name == nullptr || allocator == nullptr) {
    return MakeOrtStatus(ORT_INVALID_ARGUMENT,
                         "Invalid argument passed to the 'sessionOptions.epContextDataRead' callback.");
  }

  // ONNX Runtime does not serialize calls from different EP instances or worker threads, but the
  // JavaScript callback is user code that expects one call at a time.
  std::lock_guard<std::mutex> callLock{call_mutex_};

  if (!js_available_.load(std::memory_order_acquire)) {
    return MakeOrtStatus(ORT_FAIL, "'sessionOptions.epContextDataRead.callback' is no longer available.");
  }

  auto item = std::make_shared<CallItem>();
  item->name = name;
  item->allocator = allocator;
  item->max_data_size = max_data_size_;

  if (std::this_thread::get_id() == owner_thread_id_) {
    // ONNX Runtime can only reach this point on the JavaScript thread while that thread is inside a
    // native call made on its behalf, which is a context where N-API permits calling into JavaScript.
    // A thread-safe function call would deadlock here because the queued item could never be drained.
    InvokeOnOwnerThread(*item);
  } else {
    napi_threadsafe_function tsfn = tsfn_.load(std::memory_order_acquire);
    if (tsfn == nullptr || napi_acquire_threadsafe_function(tsfn) != napi_ok) {
      return MakeOrtStatus(ORT_FAIL, "'sessionOptions.epContextDataRead.callback' is no longer available.");
    }

    auto* payload = new std::shared_ptr<CallItem>{item};
    napi_status status = napi_call_threadsafe_function(tsfn, payload, napi_tsfn_blocking);
    if (status != napi_ok) {
      delete payload;
      napi_release_threadsafe_function(tsfn, napi_tsfn_release);
      return MakeOrtStatus(ORT_FAIL,
                           "Failed to dispatch 'sessionOptions.epContextDataRead.callback' to the JavaScript thread.");
    }

    item->Wait();
    napi_release_threadsafe_function(tsfn, napi_tsfn_release);
  }

  if (!item->succeeded) {
    return MakeOrtStatus(item->error_code, item->error_message);
  }

  *data_size = item->buffer.size;
  *buffer = item->buffer.Detach();
  return nullptr;
}

OrtStatus* ORT_API_CALL EpContextDataReadState::ReadNamedBuffer(void* state, const char* name,
                                                                OrtAllocator* allocator, void** buffer,
                                                                size_t* data_size) noexcept {
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (data_size != nullptr) {
    *data_size = 0;
  }
  if (state == nullptr) {
    return MakeOrtStatus(ORT_INVALID_ARGUMENT, "'sessionOptions.epContextDataRead' state is missing.");
  }
  return static_cast<EpContextDataReadState*>(state)->Invoke(name, allocator, buffer, data_size);
}

void ParseEpContextDataReadOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions,
                                   std::shared_ptr<EpContextDataReadState>& state) {
  Napi::Env env = options.Env();

  if (!options.Has("epContextDataRead")) {
    return;
  }
  auto value = options.Get("epContextDataRead");
  if (value.IsNull() || value.IsUndefined()) {
    return;
  }

  ORT_NAPI_THROW_TYPEERROR_IF(!value.IsObject(), env,
                              "Invalid argument: sessionOptions.epContextDataRead must be an object.");
  auto config = value.As<Napi::Object>();

  auto callbackValue = config.Get("callback");
  ORT_NAPI_THROW_TYPEERROR_IF(!callbackValue.IsFunction(), env,
                              "Invalid argument: sessionOptions.epContextDataRead.callback must be a function.");

  auto maxDataSizeValue = config.Get("maxDataSize");
  ORT_NAPI_THROW_TYPEERROR_IF(!maxDataSizeValue.IsNumber(), env,
                              "Invalid argument: sessionOptions.epContextDataRead.maxDataSize must be a number.");
  const double maxDataSize = maxDataSizeValue.As<Napi::Number>().DoubleValue();
  ORT_NAPI_THROW_RANGEERROR_IF(!std::isfinite(maxDataSize) || std::floor(maxDataSize) != maxDataSize ||
                                   maxDataSize < 1 || maxDataSize > kMaxSafeInteger ||
                                   maxDataSize >= static_cast<double>(std::numeric_limits<size_t>::max()),
                               env, "'epContextDataRead.maxDataSize' is invalid: ", maxDataSize);
  const size_t maxDataSizeBytes = static_cast<size_t>(maxDataSize);

  // Transactional setup: retain the new state first, register it next, and publish it only after the
  // native setter succeeded.
  auto newState = EpContextDataReadState::Create(env, callbackValue.As<Napi::Function>(), maxDataSizeBytes);
  try {
    sessionOptions.SetEpContextDataReadFunc(&EpContextDataReadState::ReadNamedBuffer, newState.get(),
                                            maxDataSizeBytes);
  } catch (...) {
    newState->Release();
    throw;
  }

  if (state) {
    state->Release();
  }
  state = std::move(newState);
}
