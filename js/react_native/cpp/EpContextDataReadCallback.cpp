// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "EpContextDataReadCallback.h"
#include "EpContextDataReadPolicy.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

using namespace facebook::jsi;

namespace onnxruntimejsi {

namespace {

OrtStatus* makeStatus(OrtErrorCode code, const std::string& message) noexcept {
  return Ort::GetApi().CreateStatus(code, message.c_str());
}

Value getTypedArrayProperty(Runtime& runtime,
                            const Object& typedArrayPrototype,
                            const Object& typedArray,
                            const char* propertyName) {
  auto objectConstructor =
      runtime.global().getPropertyAsObject(runtime, "Object");
  auto getOwnPropertyDescriptor = objectConstructor.getPropertyAsFunction(
      runtime, "getOwnPropertyDescriptor");
  auto descriptor =
      getOwnPropertyDescriptor
          .call(runtime, typedArrayPrototype,
                String::createFromAscii(runtime, propertyName))
          .asObject(runtime);
  auto getter = descriptor.getPropertyAsFunction(runtime, "get");
  return getter.callWithThis(runtime, typedArray);
}

bool toViewIndex(double value, size_t& result) noexcept {
  if (!std::isfinite(value) || value < 0 || std::trunc(value) != value ||
      value >= static_cast<double>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  result = static_cast<size_t>(value);
  return true;
}

}  // namespace

bool EpContextDataReadCallback::PendingCall::isFinished() noexcept {
  std::lock_guard<std::mutex> lock(mutex);
  return finished;
}

void EpContextDataReadCallback::PendingCall::finish(
    CallStatus newStatus, std::string newError,
    std::vector<uint8_t> newData) noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (finished) return;
    finished = true;
    status = newStatus;
    error = std::move(newError);
    data = std::move(newData);
  }
  cv.notify_all();
}

void EpContextDataReadCallback::PendingCall::wait() {
  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [this] { return finished; });
}

EpContextDataReadCallback::EpContextDataReadCallback(
    std::shared_ptr<Env> env, Runtime& runtime,
    std::shared_ptr<Function> callback, size_t maxDataSize)
    : env_(std::move(env)),
      maxDataSize_(maxDataSize),
      runtime_(&runtime),
      callback_(std::move(callback)) {}

EpContextDataReadCallback::~EpContextDataReadCallback() {
  invalidate();

  std::shared_ptr<Function> callback;
  {
    std::lock_guard<std::mutex> lock(stateMutex_);
    callback.swap(callback_);
  }
  if (callback && !env_->isJsThread()) {
    // Releasing a jsi::Function writes through the runtime that created it. Off the JS thread that
    // runtime may already be gone, so hand the last reference to a deliberate leak rather than
    // invalidate a pointer into freed memory. invalidate() drops the reference on the JS thread in
    // every expected teardown path, so this is a safety net rather than a routine leak.
    new std::shared_ptr<Function>(std::move(callback));
  }
}

std::shared_ptr<EpContextDataReadCallback>
EpContextDataReadCallback::createAndRegister(Runtime& runtime,
                                             const Object& options,
                                             const std::shared_ptr<Env>& env,
                                             Ort::SessionOptions& sessionOptions) {
  if (!options.hasProperty(runtime, "epContextDataRead")) {
    return nullptr;
  }
  auto value = options.getProperty(runtime, "epContextDataRead");
  if (value.isUndefined() || value.isNull()) {
    return nullptr;
  }
  if (!value.isObject()) {
    throw JSError(runtime,
                  "session option \"epContextDataRead\" must be an object");
  }
  auto config = value.asObject(runtime);

  auto callbackValue = config.getProperty(runtime, "callback");
  if (!callbackValue.isObject() ||
      !callbackValue.asObject(runtime).isFunction(runtime)) {
    throw JSError(
        runtime,
        "session option \"epContextDataRead.callback\" must be a function");
  }

  auto maxDataSizeValue = config.getProperty(runtime, "maxDataSize");
  if (!maxDataSizeValue.isNumber()) {
    throw JSError(runtime,
                  "session option \"epContextDataRead.maxDataSize\" is "
                  "required and must be a number");
  }
  auto maxDataSize = parseMaxDataSize(maxDataSizeValue.asNumber());
  if (!maxDataSize.ok) {
    throw JSError(runtime, maxDataSize.error);
  }

  auto state = std::make_shared<EpContextDataReadCallback>(
      env, runtime,
      std::make_shared<Function>(
          callbackValue.asObject(runtime).asFunction(runtime)),
      maxDataSize.value);

  // Publish the state only once ONNX Runtime accepted it. If the setter throws, `state` is
  // destroyed here on the JS thread and nothing observed a half-registered callback.
  sessionOptions.SetEpContextDataReadFunc(&EpContextDataReadCallback::read,
                                          state.get(), maxDataSize.value);

  env->addTeardownListener(state);
  return state;
}

OrtStatus* ORT_API_CALL EpContextDataReadCallback::read(
    void* state, const char* name, OrtAllocator* allocator, void** buffer,
    size_t* dataSize) noexcept {
  // Initialize the outputs before anything that can fail so ORT never observes stale values.
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (dataSize != nullptr) {
    *dataSize = 0;
  }

  if (state == nullptr || name == nullptr || allocator == nullptr ||
      buffer == nullptr || dataSize == nullptr) {
    return makeStatus(
        ORT_INVALID_ARGUMENT,
        "EPContext data read callback received an invalid argument");
  }

  try {
    return static_cast<EpContextDataReadCallback*>(state)->readImpl(
        std::string(name), allocator, buffer, dataSize);
  } catch (const std::exception& e) {
    return makeStatus(ORT_FAIL, "EPContext data read callback failed: " +
                                    std::string(e.what()));
  } catch (...) {
    return makeStatus(
        ORT_FAIL, "EPContext data read callback failed with an unknown error");
  }
}

OrtStatus* EpContextDataReadCallback::readImpl(const std::string& name,
                                               OrtAllocator* allocator,
                                               void** buffer,
                                               size_t* dataSize) {
  // ORT does not serialize calls made by different EP instances or worker threads.
  std::lock_guard<std::mutex> callLock(callMutex_);

  auto pending = std::make_shared<PendingCall>();

  if (env_->isJsThread()) {
    // Dispatching from the JS thread would deadlock, so run inline. A single state object belongs
    // to a single session load, which either runs entirely on the JS thread or entirely off it, so
    // this branch cannot overlap with a thread waiting for the JS thread to drain.
    {
      std::lock_guard<std::mutex> lock(stateMutex_);
      if (!valid_) {
        return makeStatus(ORT_FAIL,
                          "EPContext data read callback was released before "
                          "\"" +
                              name + "\" could be read");
      }
    }
    invokeOnJsThread(name, *pending);
  } else {
    {
      std::lock_guard<std::mutex> lock(stateMutex_);
      if (!valid_) {
        return makeStatus(ORT_FAIL,
                          "EPContext data read callback was released before "
                          "\"" +
                              name + "\" could be read");
      }
      pending_.push_back(pending);
    }

    std::weak_ptr<EpContextDataReadCallback> weakSelf = weak_from_this();
    const bool queued = env_->runOnJsThread([weakSelf, pending, name]() {
      if (pending->isFinished()) return;
      auto self = weakSelf.lock();
      if (!self) {
        pending->finish(CallStatus::Failed,
                        "EPContext data read callback was released before \"" +
                            name + "\" could be read");
        return;
      }
      self->invokeOnJsThread(name, *pending);
    });

    if (!queued) {
      unregisterPending(pending);
      return makeStatus(ORT_FAIL,
                        "ONNX Runtime JSI bindings were torn down before \"" +
                            name + "\" could be read");
    }

    // invalidate() finishes every registered call, so teardown cannot leave this thread blocked.
    pending->wait();
    unregisterPending(pending);
  }

  CallStatus status;
  std::string error;
  std::vector<uint8_t> data;
  {
    std::lock_guard<std::mutex> lock(pending->mutex);
    status = pending->status;
    error = std::move(pending->error);
    data = std::move(pending->data);
  }

  switch (status) {
    case CallStatus::Ok:
      break;
    case CallStatus::InvalidArgument:
      return makeStatus(ORT_INVALID_ARGUMENT, error);
    case CallStatus::Failed:
      return makeStatus(ORT_FAIL, error);
  }

  // An empty payload is reported as a null buffer, not as a zero-sized allocation.
  if (data.empty()) {
    return nullptr;
  }

  // The limit is already enforced on the JS thread; re-check before touching the allocator so the
  // guarantee holds even if the two paths ever diverge.
  auto sizeCheck = checkDataSize(data.size(), maxDataSize_, name);
  if (!sizeCheck.ok) {
    return makeStatus(ORT_INVALID_ARGUMENT, sizeCheck.error);
  }

  void* allocated = allocator->Alloc(allocator, data.size());
  if (allocated == nullptr) {
    return makeStatus(ORT_FAIL, "failed to allocate " +
                                    std::to_string(data.size()) +
                                    " bytes for EPContext data \"" + name +
                                    "\"");
  }
  std::memcpy(allocated, data.data(), data.size());
  *buffer = allocated;
  *dataSize = data.size();
  return nullptr;
}

void EpContextDataReadCallback::invokeOnJsThread(const std::string& name,
                                                 PendingCall& pending) noexcept {
  Runtime* runtime = nullptr;
  std::shared_ptr<Function> callback;
  {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!valid_ || runtime_ == nullptr || !callback_) {
      pending.finish(CallStatus::Failed,
                     "EPContext data read callback was released before \"" +
                         name + "\" could be read");
      return;
    }
    runtime = runtime_;
    callback = callback_;
  }

  try {
    auto result =
        callback->call(*runtime, String::createFromUtf8(*runtime, name));
    if (!result.isObject()) {
      pending.finish(CallStatus::InvalidArgument,
                     "session option \"epContextDataRead.callback\" must "
                     "return a Uint8Array for \"" +
                         name + "\"");
      return;
    }

    auto resultObject = result.asObject(*runtime);
    auto uint8ArrayConstructor =
        runtime->global().getPropertyAsFunction(*runtime, "Uint8Array");
    if (!resultObject.instanceOf(*runtime, uint8ArrayConstructor)) {
      pending.finish(CallStatus::InvalidArgument,
                     "session option \"epContextDataRead.callback\" must "
                     "return a Uint8Array for \"" +
                         name + "\"");
      return;
    }

    auto uint8ArrayPrototype =
        uint8ArrayConstructor.getPropertyAsObject(*runtime, "prototype");
    auto objectConstructor =
        runtime->global().getPropertyAsObject(*runtime, "Object");
    auto getPrototypeOf =
        objectConstructor.getPropertyAsFunction(*runtime, "getPrototypeOf");
    auto typedArrayPrototype =
        getPrototypeOf.call(*runtime, uint8ArrayPrototype).asObject(*runtime);

    auto arrayBufferValue = getTypedArrayProperty(
        *runtime, typedArrayPrototype, resultObject, "buffer");
    if (!arrayBufferValue.isObject() ||
        !arrayBufferValue.asObject(*runtime).isArrayBuffer(*runtime)) {
      pending.finish(
          CallStatus::InvalidArgument,
          "session option \"epContextDataRead.callback\" returned a "
          "Uint8Array with an invalid backing buffer for \"" +
              name + "\"");
      return;
    }
    auto arrayBuffer =
        arrayBufferValue.asObject(*runtime).getArrayBuffer(*runtime);

    size_t byteOffset = 0;
    size_t byteLength = 0;
    if (!toViewIndex(getTypedArrayProperty(*runtime, typedArrayPrototype,
                                           resultObject, "byteOffset")
                         .asNumber(),
                     byteOffset) ||
        !toViewIndex(getTypedArrayProperty(*runtime, typedArrayPrototype,
                                           resultObject, "byteLength")
                         .asNumber(),
                     byteLength)) {
      pending.finish(CallStatus::InvalidArgument,
                     "session option \"epContextDataRead.callback\" returned a "
                     "Uint8Array with invalid view bounds for \"" +
                         name + "\"");
      return;
    }
    if (byteOffset > arrayBuffer.size(*runtime) ||
        byteLength > arrayBuffer.size(*runtime) - byteOffset) {
      pending.finish(CallStatus::InvalidArgument,
                     "session option \"epContextDataRead.callback\" returned a "
                     "Uint8Array with an unreadable range for \"" +
                         name + "\"");
      return;
    }

    // Reject an oversized payload before copying and before any allocator call.
    auto sizeCheck = checkDataSize(byteLength, maxDataSize_, name);
    if (!sizeCheck.ok) {
      pending.finish(CallStatus::InvalidArgument, sizeCheck.error);
      return;
    }

    // Copy while still on the JS thread: the backing store belongs to JavaScript and may be
    // collected or moved once this function returns.
    std::vector<uint8_t> data;
    if (byteLength > 0) {
      const uint8_t* begin = arrayBuffer.data(*runtime) + byteOffset;
      data.assign(begin, begin + byteLength);
    }
    pending.finish(CallStatus::Ok, {}, std::move(data));
  } catch (const JSError& e) {
    pending.finish(CallStatus::Failed,
                   "session option \"epContextDataRead.callback\" threw while "
                   "reading \"" +
                       name + "\": " + e.getMessage());
  } catch (const std::exception& e) {
    pending.finish(CallStatus::Failed,
                   "session option \"epContextDataRead.callback\" failed while "
                   "reading \"" +
                       name + "\": " + std::string(e.what()));
  } catch (...) {
    pending.finish(CallStatus::Failed,
                   "session option \"epContextDataRead.callback\" failed with "
                   "an unknown error while reading \"" +
                       name + "\"");
  }
}

void EpContextDataReadCallback::unregisterPending(
    const std::shared_ptr<PendingCall>& pending) noexcept {
  std::lock_guard<std::mutex> lock(stateMutex_);
  pending_.erase(std::remove(pending_.begin(), pending_.end(), pending),
                 pending_.end());
}

void EpContextDataReadCallback::invalidate() noexcept {
  std::vector<std::shared_ptr<PendingCall>> pending;
  std::shared_ptr<Function> callback;
  const bool onJsThread = env_->isJsThread();
  {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!valid_) return;
    valid_ = false;
    runtime_ = nullptr;
    pending.swap(pending_);
    if (onJsThread) {
      // Safe to drop the JS function here: the runtime is alive on its own thread.
      callback.swap(callback_);
    }
  }

  for (auto& call : pending) {
    call->finish(CallStatus::Failed,
                 "EPContext data read callback was released while a read was "
                 "in flight");
  }
}

}  // namespace onnxruntimejsi
