// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#endif
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

enum class PoolType : uint8_t {
  kMaxPool,
  kAveragePool,
  kLpPool
};

class LpPool;

class PoolProcessContext {
 private:
  int64_t p_;

 public:
  friend class LpPool;
  PoolProcessContext() = default;
  void init(const OpKernelInfo& info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("p", &p_).IsOK());
  }
};

class AveragePool {
 public:
  static float Initialize() {
    return 0.0;
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& /*cxt*/) {
    y_data += x_data;
  }

  template <typename T>
  static void Finalize(const int64_t size, T& y_data, const PoolProcessContext& /*cxt*/) {
    y_data /= size;
  }

  static const PoolType type = PoolType::kAveragePool;
};

template <int START_VERSION>
class MaxPool;

template <>
class MaxPool<1 /*START_VERSION*/> {
 public:
  static float Initialize() {
    return std::numeric_limits<float>::lowest();
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& /*cxt*/) {
    if (x_data > y_data) {
      y_data = x_data;
    }
  }

  template <typename T>
  static void Finalize(const int64_t /*size*/, T& /*y_data*/, const PoolProcessContext& /*cxt*/) {}

  static const PoolType type = PoolType::kMaxPool;
};

template <>
class MaxPool<8 /*START_VERSION*/> {
 public:
  static const PoolType type = PoolType::kMaxPool;
};

class LpPool {
 public:
  static float Initialize() {
    return 0.0f;
  }

  template <typename T>
  static void Process(const T& x_data, T& y_data, const PoolProcessContext& cxt) {
    y_data += static_cast<T>(std::pow(std::abs(x_data), cxt.p_));
  }

  template <typename T>
  static void Finalize(const int64_t /*size*/, T& y_data, const PoolProcessContext& cxt) {
    y_data = static_cast<T>(std::pow(y_data, 1.0f / cxt.p_));
  }
  static const PoolType type = PoolType::kLpPool;
};

class PoolBase {
 private:
  static int GetStartVersion(const OpKernelInfo& info) {
    return info.node().SinceVersion();
  }

 protected:
  PoolBase(const OpKernelInfo& info)
      : op_name_(info.GetKernelDef().OpName().rfind("QLinear", 0) != 0 ? info.GetKernelDef().OpName() : info.GetKernelDef().OpName().substr(7)),
        pool_attrs_(info, op_name_, GetStartVersion(info)) {
  }

  ~PoolBase() = default;

  Status Compute(OpKernelContext* context, MLAS_POOLING_KIND kind) const;

 protected:
  const std::string op_name_;

  PoolAttributes pool_attrs_;

  inline int64_t stride_h() const {
    return pool_attrs_.global_pooling ? 1 : pool_attrs_.strides[0];
  }

  inline int64_t stride_w() const {
    return pool_attrs_.global_pooling ? 1 : pool_attrs_.strides[1];
  }

  inline int64_t stride_d() const {
    return pool_attrs_.global_pooling ? 1 : pool_attrs_.strides[2];
  }
};

}  // namespace onnxruntime
