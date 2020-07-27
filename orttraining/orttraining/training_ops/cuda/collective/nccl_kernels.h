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
};

class NcclAllGather final : public NcclKernel {
 public:
  explicit NcclAllGather(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
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
};

}  // namespace cuda
}  // namespace onnxruntime
