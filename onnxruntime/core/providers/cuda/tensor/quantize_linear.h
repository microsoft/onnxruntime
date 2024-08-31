// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <class T, class U>
class QuantizeLinear final : public CudaKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }
    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
  }

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
  int64_t block_size_;
};

template <class T, class U>
class DequantizeLinear final : public CudaKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
  }

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
};

}  // namespace cuda
}  // namespace onnxruntime
