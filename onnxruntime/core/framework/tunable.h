// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/common.h"
#ifndef SHARED_PROVIDER
#include "core/common/logging/logging.h"
#endif

namespace onnxruntime {
namespace tunable {

template <typename StreamT>
struct OpParams {
  OpParams() : stream{} {}
  explicit OpParams(StreamT stream) : stream(stream) {}
  virtual ~OpParams() = default;
  virtual std::string Signature() const = 0;
  virtual StreamT Stream() const { return stream; }
  StreamT stream;
};

template <typename StreamT>
class Timer {
 public:
  explicit Timer(StreamT stream) : stream_{stream} {}
  virtual ~Timer() = default;

  virtual void Start() = 0;
  virtual void End() = 0;

  /// Computes the elapsed time in milliseconds between Start() and End() over the specified stream.
  virtual float Duration() = 0;

 protected:
  StreamT stream_;
};

template <typename T, typename Arg, typename E = void>
struct HasIsSupportedMethod {
  constexpr static bool value = false;
};

template <typename T, typename Arg>
struct HasIsSupportedMethod<
    T, Arg, std::enable_if_t<std::is_same_v<decltype(std::declval<T>().IsSupported(std::declval<Arg>())), Status>>> {
  constexpr static bool value = true;
};

// A type erased Callable wrapper. We could have used std::function<Status<const ParamsT*>> here. However, std::function
// requires the callable object to be CopyConstructible and CopyAssignable. This is not suitable for move only functor
// or move captured lambda. So we create a simple wrapper for our purpose here.
//
// Then an Op is Status(const ParamsT*), that is, a callable accepts a const ParamsT* and returns a Status.
// This means that it can be either a free function, a functor or a lambda.
template <typename ParamsT>
class Op {
 public:
  template <typename T>
  explicit Op(T&& c) : callable_{std::make_unique<CallableImpl<T>>(std::forward<T>(c))} {}
  Op(Op&&) = default;
  Status operator()(const ParamsT* param) { return (*callable_)(param); }
  Status IsSupported(const ParamsT* param) { return (*callable_).IsSupported(param); }

 private:
  struct ICallable {
    virtual ~ICallable() = default;
    virtual Status operator()(const ParamsT*) = 0;
    virtual Status IsSupported(const ParamsT*) = 0;
  };

  template <typename T>
  struct CallableImpl : ICallable {
    explicit CallableImpl(T&& c) : c_{std::move(c)} {}
    CallableImpl(CallableImpl&&) = default;
    Status operator()(const ParamsT* param) override { return c_(param); }

    Status IsSupported(const ParamsT* param) override {
      if constexpr (HasIsSupportedMethod<T, const ParamsT*>::value) {
        return c_.IsSupported(param);
      } else {
        return c_(param);
      }
    }

   private:
    T c_;
  };

  std::unique_ptr<ICallable> callable_;
};

// NOTE: onnxruntime's Status currently does not have a StatusCode::UNSUPPORTED. Currently, we do not want to extend the
// enum. So we reuse StatusCode::INVALID_ARGUMENT for this purpose. It can be interpreted as "The input argument is not
// valid for this specialized kernel implementation.". This semantic is crucial for the tuning mechanism.
#define TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(condition, ...)  \
  do {                                                             \
    if (condition) {                                               \
      return ORT_MAKE_STATUS(NONE, INVALID_ARGUMENT, __VA_ARGS__); \
    }                                                              \
  } while (false)

template <typename ParamsT, typename TimerT>
class TunableOp {
 public:
  TunableOp() = default;
  TunableOp(TunableOp&&) = default;

  Status operator()(const ParamsT* params) {
    int id;
    if (tuning_) {
      if (kernel_map_.find(params->Signature()) == kernel_map_.end()) {
        auto maybe_proxy_params = this->PreTuning(params);
        id = FindFastest(maybe_proxy_params);
        PostTuning(maybe_proxy_params);
        kernel_map_.insert({params->Signature(), id});
      } else {
        id = kernel_map_[params->Signature()];
      }
    } else {
      id = default_id_;
    }
    ORT_RETURN_IF_ERROR(ops_[id](params));
    return Status::OK();
  }

  void EnableTuning() {
    tuning_ = true;
    for (auto nested_op_ptr : nested_tunable_ops_) {
      nested_op_ptr->EnableTuning();
    }
  }

  void DisableTuning() {
    tuning_ = false;
    for (auto nested_op_ptr : nested_tunable_ops_) {
      nested_op_ptr->DisableTuning();
    }
  }

  // We might want to do some tricks to the `params`, e.g., some op will use a buffer for input and output at the same
  // time, so it will do inplace update to it. If we blindly tune over the `params`, there will be accumulated update
  // to that buffer during FindFastest, which is an undesired side effect. In this case, we must prepare a new (proxy)
  // params struct for the tuning to avoid this side effect.
  virtual const ParamsT* PreTuning(const ParamsT* params) {
    return params;
  }

  virtual void PostTuning(const ParamsT* /*params*/) {
    // Do nothing if we are not playing around with params
  }

  virtual ~TunableOp() = default;

 protected:
  // set the default op to be used in non-tuning scenario
  void SetDefaultId(int id) {
    ORT_ENFORCE(id < static_cast<int>(ops_.size()), "TunableOp id out of bound");
    default_id_ = id;
  }

  void RegisterNestedTunableOp(TunableOp<ParamsT, TimerT>* op_ptr) {
    nested_tunable_ops_.insert(op_ptr);
    if (tuning_) {
      op_ptr->EnableTuning();
    } else {
      op_ptr->DisableTuning();
    }

    // Add an op for this tunable op as well.
    ops_.emplace_back([op_ptr](const ParamsT* params) {
      return op_ptr->operator()(params);
    });
  }

 private:
  static void WarmUp(Op<ParamsT>& op, const ParamsT* param) {
    constexpr const int num_iter = 4;
    for (int i = 0; i < num_iter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
  }

  static double Profile(Op<ParamsT>& op, const ParamsT* param) {
    constexpr const int num_iter = 100;
    TimerT timer{param->Stream()};
    timer.Start();
    for (int i = 0; i < num_iter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
    timer.End();
    return timer.Duration() / num_iter;
  }

  static bool IsSupported(Op<ParamsT>& op, const ParamsT* param) {
    Status status = op.IsSupported(param);
    if (status.Category() == common::StatusCategory::NONE && status.Code() == common::StatusCode::INVALID_ARGUMENT) {
      return false;
    }
    ORT_THROW_IF_ERROR(status);
    return true;
  }

  std::string OpSignature() const {
#ifdef ORT_NO_RTTI
    ORT_THROW("TunableOp must be built with RTTI enabled");
#else
#ifndef _WIN32
    const auto* name = typeid(*this).name();
    char buf[256];
    size_t buf_len = 256;
    abi::__cxa_demangle(name, buf, &buf_len, nullptr);
    buf[255] = '\0';
    return buf;
#else
    return typeid(*this).name();
#endif
#endif
  }

 protected:
  virtual int FindFastest(const ParamsT* params) {
    return FindFastestImpl(params, ops_);
  }

  int FindFastestImpl(const ParamsT* params, const std::vector<Op<ParamsT>>& candidates) {
    auto op_sig = OpSignature();
    auto param_sig = params->Signature();
    LOGS_DEFAULT(VERBOSE) << "FindFastestImpl for " << op_sig << '(' << param_sig << ')';
    auto min_time = std::numeric_limits<double>::infinity();
    int id = -1;

    for (size_t i = 0; i < candidates.size(); i++) {
      auto& candidate = const_cast<Op<ParamsT>&>(candidates[i]);
      if (!IsSupported(candidate, params)) {
        LOGS_DEFAULT(VERBOSE) << "FindFastestImpl found unsupported " << op_sig
                              << '(' << param_sig << ") id=" << i;
        continue;
      }

      WarmUp(candidate, params);
      auto time = Profile(candidate, params);
      if (time < min_time) {
        min_time = time;
        id = static_cast<int>(i);
      }
    }
    ORT_ENFORCE(id >= 0, "Cannot found viable op");
    LOGS_DEFAULT(VERBOSE) << "FindFastestImpl for " << op_sig << '(' << param_sig << ") found fastest with id=" << id;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return id;
  }

  std::vector<Op<ParamsT>> ops_;

 private:
  // mapping from Signature to best impl
  std::unordered_map<std::string, int> kernel_map_;
  // the default impl to use when tuning is disabled
  int default_id_{0};
  bool tuning_{false};
  // Registered tunable sub-ops for nested tuning
  std::unordered_set<TunableOp<ParamsT, TimerT>*> nested_tunable_ops_;
};

}  // namespace tunable
}  // namespace onnxruntime
