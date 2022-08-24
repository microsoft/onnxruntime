// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "contrib_ops/rocm/bert/util.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

struct OpParams {
  OpParams() : stream{} {}
  explicit OpParams(hipStream_t stream) : stream(stream) {}
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
        id = FindFastest(params);
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
    Timer timer{};
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

  int FindFastest(const ParamsT* params) {
    auto min_time = std::numeric_limits<double>::infinity();
    int id = -1;
    for (size_t i = 0; i < this->ops_.size(); i++) {
      if (!IsSupported(ops_[i], params)) {
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

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
