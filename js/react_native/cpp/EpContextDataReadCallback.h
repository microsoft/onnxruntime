// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Env.h"
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include "onnxruntime_cxx_api.h"
#include <string>
#include <vector>

namespace onnxruntimejsi {

/**
 * @brief Bridges OrtReadNamedBufferFunc to the `epContextDataRead.callback` JavaScript function.
 *
 * ONNX Runtime may call the read function from any thread while a session is being created and
 * does not serialize calls made by different EP instances, so every invocation is serialized here
 * and marshalled to the JS thread unless the caller demonstrably runs on it already.
 *
 * An instance owns a strong reference to the JS function for the whole session lifetime and must
 * outlive the Ort::Session it was registered with.
 */
class EpContextDataReadCallback
    : public EnvTeardownListener,
      public std::enable_shared_from_this<EpContextDataReadCallback> {
 public:
  EpContextDataReadCallback(std::shared_ptr<Env> env,
                            facebook::jsi::Runtime& runtime,
                            std::shared_ptr<facebook::jsi::Function> callback,
                            size_t maxDataSize);

  ~EpContextDataReadCallback() override;

  EpContextDataReadCallback(const EpContextDataReadCallback&) = delete;
  EpContextDataReadCallback& operator=(const EpContextDataReadCallback&) = delete;
  EpContextDataReadCallback(EpContextDataReadCallback&&) = delete;
  EpContextDataReadCallback& operator=(EpContextDataReadCallback&&) = delete;

  /**
   * @brief Parse `epContextDataRead` and register it with `sessionOptions`.
   *
   * Must be called on the JS thread. Returns nullptr when the option is absent, undefined or null,
   * and throws facebook::jsi::JSError when it is present but malformed.
   *
   * Registration is transactional: the state is returned to the caller only after ONNX Runtime
   * accepted it. The caller must keep the returned state alive for at least as long as the session
   * created from `sessionOptions`.
   */
  static std::shared_ptr<EpContextDataReadCallback>
  createAndRegister(facebook::jsi::Runtime& runtime,
                    const facebook::jsi::Object& options,
                    const std::shared_ptr<Env>& env,
                    Ort::SessionOptions& sessionOptions);

  /**
   * @brief OrtReadNamedBufferFunc entry point. No exception ever crosses the C ABI.
   */
  static OrtStatus* ORT_API_CALL read(void* state, const char* name,
                                      OrtAllocator* allocator, void** buffer,
                                      size_t* dataSize) noexcept;

  /**
   * @brief Stop serving reads and wake every thread blocked on the JS thread.
   *
   * Callable from any thread and more than once. Subsequent reads fail instead of falling back to
   * any other data source.
   */
  void invalidate() noexcept;

  void onEnvTeardown() noexcept override { invalidate(); }

  size_t maxDataSize() const noexcept { return maxDataSize_; }

 private:
  enum class CallStatus {
    Ok,
    // Bad callback result: wrong type, unreadable buffer, or oversized payload.
    InvalidArgument,
    // The callback threw, or the bindings were torn down mid-call.
    Failed,
  };

  struct PendingCall {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    CallStatus status = CallStatus::Failed;
    std::string error;
    std::vector<uint8_t> data;

    bool isFinished() noexcept;
    void finish(CallStatus status, std::string error,
                std::vector<uint8_t> data = {}) noexcept;
    void wait();
  };

  OrtStatus* readImpl(const std::string& name, OrtAllocator* allocator,
                      void** buffer, size_t* dataSize);

  // Runs on the JS thread. Fills `pending` with the callback outcome.
  void invokeOnJsThread(const std::string& name, PendingCall& pending) noexcept;

  void unregisterPending(const std::shared_ptr<PendingCall>& pending) noexcept;

  const std::shared_ptr<Env> env_;
  const size_t maxDataSize_;

  // Serializes read callbacks so the JS function never observes overlapping calls.
  std::mutex callMutex_;

  // Guards the members below. Never held while blocking, so invalidate() cannot deadlock against
  // a read that is waiting on the JS thread.
  std::mutex stateMutex_;
  bool valid_ = true;
  facebook::jsi::Runtime* runtime_;
  std::shared_ptr<facebook::jsi::Function> callback_;
  std::vector<std::shared_ptr<PendingCall>> pending_;
};

}  // namespace onnxruntimejsi
