#pragma once

#include <ReactCommon/CallInvoker.h>
#include <algorithm>
#include <functional>
#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include "onnxruntime_cxx_api.h"
#include <thread>
#include <vector>

namespace onnxruntimejsi {

/**
 * @brief Notified once the host tears the JSI bindings down.
 *
 * Teardown is not guaranteed to happen on the JS thread, so listeners must not touch the runtime.
 */
class EnvTeardownListener {
 public:
  virtual ~EnvTeardownListener() = default;

  virtual void onEnvTeardown() noexcept = 0;
};

class Env : public std::enable_shared_from_this<Env> {
 public:
  // Constructed from install(), which the host always calls on the JS thread.
  Env(std::shared_ptr<facebook::react::CallInvoker> jsInvoker)
      : jsInvoker_(jsInvoker), jsThreadId_(std::this_thread::get_id()) {}

  ~Env() { invalidate(); }

  inline void initOrtEnv(OrtLoggingLevel logLevel, const char* logid) {
    if (ortEnv_) {
      return;
    }
    ortEnv_ = std::make_shared<Ort::Env>(logLevel, logid);
  }

  inline void setTensorConstructor(
      std::shared_ptr<facebook::jsi::WeakObject> tensorConstructor) {
    tensorConstructor_ = tensorConstructor;
  }

  inline facebook::jsi::Value
  getTensorConstructor(facebook::jsi::Runtime& runtime) const {
    return tensorConstructor_->lock(runtime);
  }

  inline Ort::Env& getOrtEnv() const { return *ortEnv_; }

  /**
   * @brief Whether the caller runs on the thread that installed the bindings, i.e. the JS thread.
   */
  inline bool isJsThread() const noexcept {
    return std::this_thread::get_id() == jsThreadId_;
  }

  /**
   * @brief Queue work on the JS thread.
   *
   * @return false when the bindings were torn down and the work will never run.
   */
  inline bool runOnJsThread(std::function<void()>&& func) {
    std::shared_ptr<facebook::react::CallInvoker> jsInvoker;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jsInvoker = jsInvoker_;
    }
    if (!jsInvoker) return false;
    jsInvoker->invokeAsync(std::move(func));
    return true;
  }

  inline void
  addTeardownListener(const std::weak_ptr<EnvTeardownListener>& listener) {
    std::lock_guard<std::mutex> lock(mutex_);
    teardownListeners_.erase(
        std::remove_if(teardownListeners_.begin(), teardownListeners_.end(),
                       [](const std::weak_ptr<EnvTeardownListener>& entry) {
                         return entry.expired();
                       }),
        teardownListeners_.end());
    teardownListeners_.push_back(listener);
  }

  /**
   * @brief Stop dispatching to the JS thread and notify teardown listeners.
   *
   * Callable from any thread and more than once. Never touches the runtime, so it is safe to call
   * from a host teardown hook that does not run on the JS thread.
   */
  inline void invalidate() noexcept {
    std::vector<std::shared_ptr<EnvTeardownListener>> listeners;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!jsInvoker_ && teardownListeners_.empty()) return;
      for (auto& entry : teardownListeners_) {
        if (auto listener = entry.lock()) {
          listeners.push_back(std::move(listener));
        }
      }
      teardownListeners_.clear();
      jsInvoker_.reset();
    }
    for (auto& listener : listeners) {
      listener->onEnvTeardown();
    }
  }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<facebook::react::CallInvoker> jsInvoker_;
  std::vector<std::weak_ptr<EnvTeardownListener>> teardownListeners_;
  const std::thread::id jsThreadId_;
  std::shared_ptr<facebook::jsi::WeakObject> tensorConstructor_;
  std::shared_ptr<Ort::Env> ortEnv_;
};

}  // namespace onnxruntimejsi
