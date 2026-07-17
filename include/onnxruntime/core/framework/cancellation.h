// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

// Portable cancellation primitive that mirrors the subset of the C++20
// <stop_token> API that ONNX Runtime relies on (stop_token / stop_source /
// stop_callback). std::stop_token is not available on every toolchain ONNX
// Runtime targets (for example the Android NDK libc++ and the libc++ shipped
// with Xcode < 26 require -fexperimental-library and even then fail to link
// libc++experimental), so cancellation is expressed with these types instead.
// Only <atomic>, <mutex>, <condition_variable>, <thread> and <memory> are used,
// which are available on all supported platforms.

namespace onnxruntime {

class CancellationToken;
class CancellationSource;
template <class Callback>
class CancellationCallback;

namespace cancellation_detail {

// Type-erased node stored in the intrusive callback list of a CancellationState.
struct CancellationCallbackNode {
  CancellationCallbackNode* prev = nullptr;
  CancellationCallbackNode* next = nullptr;
  bool linked = false;

  virtual void Invoke() noexcept = 0;

 protected:
  ~CancellationCallbackNode() = default;
};

// Shared control block referenced by a source and all of its tokens/callbacks.
// Cancellation is a one-way latch: once requested it stays requested. Registered
// callbacks fire exactly once, either on the thread that requests cancellation
// or, when cancellation was already requested at registration time, on the
// registering thread.
class CancellationState {
 public:
  bool StopRequested() const noexcept {
    return stop_requested_.load(std::memory_order_acquire);
  }

  // Returns true only for the call that performs the not-stopped -> stopped
  // transition. Registered callbacks are invoked with the lock released.
  bool RequestStop() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_requested_.load(std::memory_order_relaxed)) {
      return false;
    }
    stop_requested_.store(true, std::memory_order_release);

    invoking_thread_ = std::this_thread::get_id();
    while (head_ != nullptr) {
      CancellationCallbackNode* node = head_;
      head_ = node->next;
      if (head_ != nullptr) {
        head_->prev = nullptr;
      }
      node->prev = nullptr;
      node->next = nullptr;
      node->linked = false;

      invoking_node_ = node;
      lock.unlock();
      node->Invoke();
      lock.lock();
      invoking_node_ = nullptr;
      finished_.notify_all();
    }
    return true;
  }

  // Registers a callback node. If cancellation has already been requested the
  // callback runs immediately on the calling thread and the node is not linked.
  void Register(CancellationCallbackNode* node) noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_requested_.load(std::memory_order_relaxed)) {
      lock.unlock();
      node->Invoke();
      return;
    }
    node->prev = nullptr;
    node->next = head_;
    if (head_ != nullptr) {
      head_->prev = node;
    }
    head_ = node;
    node->linked = true;
  }

  // Removes a callback node. If the node is currently being invoked on another
  // thread, blocks until that invocation finishes so the callback's captured
  // state stays alive. Safe to call from within the node's own callback.
  void Deregister(CancellationCallbackNode* node) noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (node->linked) {
      if (node->prev != nullptr) {
        node->prev->next = node->next;
      } else {
        head_ = node->next;
      }
      if (node->next != nullptr) {
        node->next->prev = node->prev;
      }
      node->linked = false;
      return;
    }

    if (invoking_node_ == node && invoking_thread_ != std::this_thread::get_id()) {
      finished_.wait(lock, [this, node]() noexcept { return invoking_node_ != node; });
    }
  }

 private:
  std::atomic<bool> stop_requested_{false};
  std::mutex mutex_;
  std::condition_variable finished_;
  CancellationCallbackNode* head_ = nullptr;
  CancellationCallbackNode* invoking_node_ = nullptr;
  std::thread::id invoking_thread_{};
};

}  // namespace cancellation_detail

// Copyable handle used to observe cancellation, equivalent to std::stop_token.
// A default-constructed token has no associated state and is never cancellable.
class CancellationToken {
 public:
  CancellationToken() noexcept = default;

  bool stop_requested() const noexcept {
    return state_ != nullptr && state_->StopRequested();
  }

  bool stop_possible() const noexcept {
    return state_ != nullptr;
  }

  friend bool operator==(const CancellationToken& lhs, const CancellationToken& rhs) noexcept {
    return lhs.state_ == rhs.state_;
  }

 private:
  friend class CancellationSource;
  template <class Callback>
  friend class CancellationCallback;

  explicit CancellationToken(std::shared_ptr<cancellation_detail::CancellationState> state) noexcept
      : state_{std::move(state)} {}

  std::shared_ptr<cancellation_detail::CancellationState> state_;
};

// Owns the cancellation state and requests cancellation, like std::stop_source.
// Copies share the same underlying state.
class CancellationSource {
 public:
  CancellationSource()
      : state_{std::make_shared<cancellation_detail::CancellationState>()} {}

  CancellationToken get_token() const noexcept {
    return CancellationToken{state_};
  }

  // Returns true only for the call that performs the stop transition.
  bool request_stop() noexcept {
    return state_ != nullptr && state_->RequestStop();
  }

  bool stop_requested() const noexcept {
    return state_ != nullptr && state_->StopRequested();
  }

 private:
  std::shared_ptr<cancellation_detail::CancellationState> state_;
};

// RAII registration of a callback that runs when cancellation is requested on
// the associated token (or immediately if it has already been requested),
// equivalent to std::stop_callback. The destructor removes the registration and
// blocks if the callback is concurrently running on another thread.
template <class Callback>
class CancellationCallback final : private cancellation_detail::CancellationCallbackNode {
 public:
  template <class C>
  CancellationCallback(const CancellationToken& token, C&& callback)
      : callback_{std::forward<C>(callback)}, state_{token.state_} {
    if (state_ != nullptr) {
      state_->Register(this);
    }
  }

  ~CancellationCallback() {
    if (state_ != nullptr) {
      state_->Deregister(this);
    }
  }

  CancellationCallback(const CancellationCallback&) = delete;
  CancellationCallback& operator=(const CancellationCallback&) = delete;
  CancellationCallback(CancellationCallback&&) = delete;
  CancellationCallback& operator=(CancellationCallback&&) = delete;

 private:
  void Invoke() noexcept override {
    callback_();
  }

  Callback callback_;
  std::shared_ptr<cancellation_detail::CancellationState> state_;
};

template <class Callback>
CancellationCallback(CancellationToken, Callback) -> CancellationCallback<Callback>;

}  // namespace onnxruntime
