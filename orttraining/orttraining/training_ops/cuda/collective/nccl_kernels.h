// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "nccl_common.h"

namespace onnxruntime {
namespace cuda {

class NcclAllReduce final : public NcclKernel {
 public:
  explicit NcclAllReduce(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_input_readies_;
};

class NcclAllGather final : public NcclKernel {
 public:
  explicit NcclAllGather(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
  private:
  int64_t partition_lb_;
  int64_t partition_ub_;
  int64_t max_group_size_;
  bool partition_even_;
};

class NcclReduceScatter final : public NcclKernel {
 public:
  explicit NcclReduceScatter(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

class NcclReduce final : public NcclKernel {
 public:
  explicit NcclReduce(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t root_rank_;
  int64_t num_input_readies_;
};

}  // namespace cuda
}  // namespace onnxruntime
