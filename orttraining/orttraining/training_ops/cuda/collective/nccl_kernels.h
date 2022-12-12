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

/*
 * Defines new version of Nccl classes that independent with training::DistributedContext
 * only rely on MPI
 */
class NcclAllReduceV2 final : public NcclKernelV2 {
 public:
  explicit NcclAllReduceV2(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

class NcclAllGatherV2 final : public NcclKernelV2 {
 public:
  explicit NcclAllGatherV2(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  int64_t world_size_;
};

}  // namespace cuda
}  // namespace onnxruntime
