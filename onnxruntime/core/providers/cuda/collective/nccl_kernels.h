// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/collective/nccl_common.h"

namespace onnxruntime {
namespace cuda {

class NcclAllReduce final : public NcclKernel {
 public:
  explicit NcclAllReduce(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_tensor_fusion_ = false;
};

class NcclAllGather final : public NcclKernel {
 public:
  explicit NcclAllGather(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  size_t AllGatherCount(const TensorShape& input_shape) const;
  size_t BroadcastCount(const TensorShape& input_shape) const;

  bool use_tensor_fusion_ = false;
};

class NcclReduceScatter final : public NcclKernel {
 public:
  explicit NcclReduceScatter(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  TensorShape OutputShape(const TensorShape& input_shape) const;
  size_t ReduceScatterCount(const TensorShape& input_shape) const;
  size_t ReduceCount(const TensorShape& input_shape) const;

  bool use_tensor_fusion_ = false;
};

}  // namespace cuda
}  // namespace onnxruntime
