// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cxxabi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/util.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

struct OpParams {
  OpParams() : stream{} {}
  explicit OpParams(hipStream_t stream) : stream(stream) {}
  virtual ~OpParams() = default;
  virtual std::string Signature() const = 0;
  hipStream_t stream;
};

// A type erased Callable wrapper. We could have used std::function<Status<const ParamT*>> here. However, std::function
// requires the callable object to be CopyConstructible and CopyAssignable. This is not suitable for move only functor
// or move captured lambda. So we create a simple wrapper for our purpose here.
//
// Then an Op is Status(const ParamT*), that is, a callable accepts a const ParamT* and returns a Status.
// This means that it can be either a free function, a functor or a lambda.
template <typename ParamT>
class Op {
 public:
  template <typename T>
  explicit Op(T&& c) : callable_{std::make_unique<CallableImpl<T>>(std::forward<T>(c))} {}
  Status operator()(const ParamT* param) { return (*callable_)(param); }

 private:
  struct ICallbale {
    virtual ~ICallbale() = default;
    virtual Status operator()(const ParamT*) = 0;
  };

  template <typename T>
  struct CallableImpl : ICallbale {
    explicit CallableImpl(T&& c) : c_{std::move(c)} {}
    Status operator()(const ParamT* param) override { return c_(param); }

   private:
    T c_;
  };

  std::unique_ptr<ICallbale> callable_;
};

// NOTE: onnxruntime's Status currently does not have a StatusCode::UNSUPPORTED. Currently, we do not want to extend the
// enum. So we reuse StatusCode::INVALID_ARGUMENT for this purpose. It can be interpreted as "The input argument is not
// valid for this specialized kernel implementation.". This semantic is crucial for the tuning mechanism.
#define TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(condition, ...)   \
  do {                                                             \
    if (condition) {                                               \
      return ORT_MAKE_STATUS(NONE, INVALID_ARGUMENT, __VA_ARGS__); \
    }                                                              \
  } while (false)

template <typename ParamsT>
class TunableOp {
 public:
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
  }

  void DisableTuning() {
    tuning_ = false;
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
    ORT_ENFORCE(id < ops_.size(), "TunableOp id out of bound");
    default_id_ = id;
  }

 private:
  static void WarmUp(Op<ParamsT>& op, const ParamsT* param) {
    const int num_iter = 4;
    for (int i = 0; i < num_iter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
  }

  static double Profile(Op<ParamsT>& op, const ParamsT* param) {
    const int num_iter = 100;
    Timer timer{param->stream};
    timer.Start();
    for (int i = 0; i < num_iter; i++) {
      ORT_THROW_IF_ERROR(op(param));
    }
    timer.End();
    return timer.Duration() / num_iter;
  }

  static bool IsSupported(Op<ParamsT>& op, const ParamsT* param) {
    Status status = op(param);
    if (status.Category() == common::StatusCategory::NONE && status.Code() == common::StatusCode::INVALID_ARGUMENT) {
      return false;
    }
    ORT_THROW_IF_ERROR(status);
    return true;
  }

  std::string OpSignature() const {
    const auto* name = typeid(*this).name();
    char buf[256];
    size_t buf_len = 256;
    abi::__cxa_demangle(name, buf, &buf_len, nullptr);
    buf[255] = '\0';
    return buf;
  }

  int FindFastest(const ParamsT* params) {
    auto op_sig = OpSignature();
    auto param_sig = params->Signature();
    LOGS_DEFAULT(VERBOSE) << "FindFastest for " << op_sig << '(' << param_sig << ')';
    auto min_time = std::numeric_limits<double>::infinity();
    int id = -1;
    for (size_t i = 0; i < this->ops_.size(); i++) {
      if (!IsSupported(ops_[i], params)) {
        LOGS_DEFAULT(VERBOSE) << "FindFastest found unsupported " << op_sig << '(' << param_sig << ") id=" << i;
        continue;
      }

      WarmUp(ops_[i], params);
      auto time = Profile(ops_[i], params);
      if (time < min_time) {
        min_time = time;
        id = static_cast<int>(i);
      }
    }
    ORT_ENFORCE(id >= 0, "Cannot found viable op");
    LOGS_DEFAULT(VERBOSE) << "FindFastest for " << op_sig << '(' << param_sig << ") found fastest with id=" << id;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return id;
  }

 protected:
  std::vector<Op<ParamsT>> ops_;

 private:
  // mapping from Signature to best impl
  std::unordered_map<std::string, int> kernel_map_;

  // the default impl to use when tuning is disabled
  int default_id_{0};

  bool tuning_{false};
};

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
