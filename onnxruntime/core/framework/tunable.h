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
#include "core/framework/execution_provider.h"
#include "core/framework/stream_handles.h"
#include "core/framework/tuning_context.h"

namespace onnxruntime {

template <typename TuningContextT, typename NativeStreamT>
struct OpParams {
  OpParams() : tuning_ctx{nullptr}, stream{} {}
  OpParams(TuningContextT* tuning_ctx, Stream* stream) : tuning_ctx(tuning_ctx), stream(stream) {}
  virtual ~OpParams() = default;
  virtual std::string Signature() const = 0;
  inline onnxruntime::Stream* Stream() const { return stream; }
  inline TuningContextT* TuningContext() const { return tuning_ctx; }
  inline NativeStreamT StreamHandle() const {
    return nullptr != stream ? static_cast<NativeStreamT>(stream->GetHandle()) : nullptr;
  }

  // NOTE: the reason of TuningContext does not contains the Stream is that ORT now supports multiple stream and the
  // stream may change from call to call.
  TuningContextT* tuning_ctx;
  onnxruntime::Stream* stream;
};

template <typename StreamT>
class ITimer {
 public:
  using NativeStreamT = StreamT;

  explicit ITimer(StreamT stream) : stream_{stream} {}
  virtual ~ITimer() = default;

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
  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<Op<ParamsT>, std::remove_cv_t<std::remove_reference_t<T>>>,
                            void>>
  Op(T&& c) : callable_{std::make_unique<CallableImpl<T>>(std::forward<T>(c))} {}  // NOLINT(google-explicit-constructor)
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
#define TUNABLE_OP_UNSUPPORTED(...) ORT_MAKE_STATUS(NONE, INVALID_ARGUMENT, __VA_ARGS__)
#define TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(condition, ...) \
  do {                                                            \
    if (condition) {                                              \
      return TUNABLE_OP_UNSUPPORTED(__VA_ARGS__);                 \
    }                                                             \
  } while (false)

template <typename ParamsT, typename TimerT>
class TunableOp {
 public:
  TunableOp() = default;
  TunableOp(TunableOp&&) = default;
  virtual ~TunableOp() = default;

  Status operator()(const ParamsT* params) {
    int id = -1;
    ITuningContext* ctx = params->TuningContext();
    if (ctx->IsTunableOpEnabled()) {
      auto& mgr = ctx->GetTuningResultsManager();
      auto op_sig = Signature();
      auto params_sig = params->Signature();

      // Usage is enabled, then we are free to use previous tuning result.
      id = mgr.Lookup(op_sig, params_sig);
      if (id > static_cast<int>(ops_.size())) {
        LOGS_DEFAULT(ERROR) << "Invalid TunableOp kernel id for " << op_sig
                            << ", id:" << id << ", registered op:" << ops_.size();
        mgr.Delete(op_sig, params_sig);
        id = -1;
      }

      // If there is not previous tuning result been found, we do the tuning iff tuning is enabled
      if (id < 0 && ctx->IsTuningEnabled()) {
        auto maybe_proxy_params = PreTuning(params);
        id = FindFastest(maybe_proxy_params);
        PostTuning(maybe_proxy_params);
        mgr.Add(op_sig, params_sig, id);
      }
    }
    ORT_RETURN_IF_ERROR(ops_[id < 0 ? default_id_ : id](params));
    return Status::OK();
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

  std::string Signature() {
    // According to C++17 standard https://wg21.link/n4659 section 15.7.4
    // > if the operand of typeid refers to the
    // > object under construction or destruction, typeid yields the std::type_info object representing the constructor
    // > or destructor’s class.
    // So delay the op signature generation. See https://github.com/microsoft/onnxruntime/pull/14709
    std::call_once(signature_init_once_, [this]() { signature_ = CreateSignature(); });
    return signature_;
  }

 protected:
  // set the default op to be used in non-tuning scenario
  void SetDefaultId(int id) {
    ORT_ENFORCE(id < static_cast<int>(ops_.size()), "TunableOp id out of bound");
    default_id_ = id;
  }

  void RegisterOp(Op<ParamsT>&& op) {
    this->ops_.emplace_back(std::move(op));
  }

  int NumberOfOps() {
    return this->ops_.size();
  }

  void RegisterNestedTunableOp(TunableOp<ParamsT, TimerT>* op_ptr) {
    nested_tunable_ops_.insert(op_ptr);

    // Add an op for this tunable op as well.
    RegisterOp([op_ptr](const ParamsT* params) {
      return op_ptr->operator()(params);
    });
  }

 private:
  static void WarmUp(Op<ParamsT>& op, const ParamsT* param) {
    constexpr const int num_iter = 1;
    for (int i = 0; i < num_iter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
  }

  static double Profile(Op<ParamsT>& op, const ParamsT* param, int num_iter) {
    TimerT timer{param->StreamHandle()};
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
      LOGS_DEFAULT(VERBOSE) << "unsupported reason: " << status.ErrorMessage();
      return false;
    }
    ORT_THROW_IF_ERROR(status);
    return true;
  }

 protected:
  virtual int FindFastest(const ParamsT* params) {
    return FindFastestImpl(params, ops_);
  }

  int FindFastestImpl(const ParamsT* params, const std::vector<Op<ParamsT>>& candidates) {
    ITuningContext* ctx = params->TuningContext();
    auto op_sig = Signature();
    auto param_sig = params->Signature();
    LOGS_DEFAULT(VERBOSE) << "FindFastestImpl for " << op_sig << '(' << param_sig << ')';
    auto min_time = std::numeric_limits<double>::infinity();
    int id = -1;

    constexpr const int max_tuning_iter = 100;
    constexpr const int approx_num_iter = 3;

    for (size_t i = 0; i < candidates.size(); i++) {
      auto& candidate = const_cast<Op<ParamsT>&>(candidates[i]);
      if (!IsSupported(candidate, params)) {
        LOGS_DEFAULT(VERBOSE) << "FindFastestImpl found unsupported " << op_sig << '(' << param_sig << ") id=" << i;
        continue;
      }

      WarmUp(candidate, params);

      auto approx_duration = Profile(candidate, params, approx_num_iter);
      if (approx_duration > 2 * min_time) {
        LOGS_DEFAULT(VERBOSE) << "FindFastestImpl skip slow instance " << op_sig << '(' << param_sig << ") id=" << i;
        continue;
      }
      int tuning_iter = std::max(1, int(std::min(double(max_tuning_iter), ctx->GetMaxTuningDurationMs() / approx_duration)));

      LOGS_DEFAULT(VERBOSE) << "FindFastestImpl run instance " << op_sig << '(' << param_sig << ") id=" << i << " " << tuning_iter << " times.";

      auto time = Profile(candidate, params, tuning_iter);
      if (time < min_time) {
        min_time = time;
        id = static_cast<int>(i);
      }
    }
    ORT_ENFORCE(id >= 0, "Could not find viable op");
    LOGS_DEFAULT(VERBOSE) << "FindFastestImpl for " << op_sig << '(' << param_sig << ") found fastest with id=" << id;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return id;
  }

 private:
  std::string CreateSignature() {
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

  mutable std::once_flag signature_init_once_;
  std::string signature_;

  // the default impl to use when tuning is disabled
  int default_id_{0};

  std::vector<Op<ParamsT>> ops_;

  // Registered tunable sub-ops for nested tuning
  std::unordered_set<TunableOp<ParamsT, TimerT>*> nested_tunable_ops_;
};

}  // namespace onnxruntime
