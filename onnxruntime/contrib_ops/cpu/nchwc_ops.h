// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_impl.h"
#include "core/providers/cpu/nn/pool.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class ReorderInput : public OpKernel {
 public:
  ReorderInput(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class ReorderOutput : public OpKernel {
 public:
  ReorderOutput(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("channels", &channels_).IsOK());
    ORT_ENFORCE(channels_ > 0, "invalid channel count");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t channels_;
};

template <typename T>
class NchwcConv : public Conv<T> {
 public:
  NchwcConv(const OpKernelInfo& info) : Conv<T>(info) {
    Conv<T>::activation_ = info.GetAttrOrDefault<std::string>("activation", "");
    Conv<T>::alpha_ = info.GetAttrOrDefault("alpha", 0.01f);
  }

  Status Compute(OpKernelContext* context) const override;
};

class NchwcPoolBase : public OpKernel, public PoolBase {
 public:
  NchwcPoolBase(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
  }

  Status Compute(OpKernelContext* context, MLAS_POOLING_KIND kind) const;
};

class NchwcMaxPool : public NchwcPoolBase {
 public:
  NchwcMaxPool(const OpKernelInfo& info) : NchwcPoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class NchwcAveragePool : public NchwcPoolBase {
 public:
  NchwcAveragePool(const OpKernelInfo& info) : NchwcPoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
