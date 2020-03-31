// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {

template <typename T, typename PoolType>
class Pool : public OpKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
    const std::string& op_name = info.GetKernelDef().OpName();
    if (op_name == "LpPool" || op_name == "GlobalLpPool") {
      pool_context_.init(info);
    }
  }

  ~Pool() override = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolProcessContext pool_context_;
};

class PoolVersion12 : public OpKernel, public PoolBase {
 public:
  explicit PoolVersion12(const OpKernelInfo& info) : OpKernel(info), PoolBase(info), pool_context_() {
    const std::string& op_name = info.GetKernelDef().OpName();
    if (op_name == "LpPool" || op_name == "GlobalLpPool") {
      pool_context_.init(info);
    }
  }

  ~PoolVersion12() override = default;
  ;

  Status Compute(OpKernelContext* context) const override;

 private:
  using PoolType = MaxPool<8>;

  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  template <typename T>
  struct ComputeHelper {
    Status operator()(const PoolVersion12* inst, OpKernelContext* context) const {
      return inst->ComputeImpl<T>(context);
    }
  };

  PoolProcessContext pool_context_;
};  // namespace onnxruntime

}  // namespace onnxruntime
