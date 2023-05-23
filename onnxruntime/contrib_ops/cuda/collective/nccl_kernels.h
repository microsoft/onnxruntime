// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#if defined(ORT_USE_NCCL)
#include <nccl.h>
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

// -----------------------------------------------------------------------
// Defines a new version of nccl classes
// that independent with training::DistributedRunContext, only rely on MPI
// -----------------------------------------------------------------------
class NcclContext final {
 public:
  NcclContext();
  ~NcclContext();

  ncclComm_t Comm() {
    return comm_;
  }

  int Rank() const {
    return rank_;
  }

  int Size() const {
    return world_size_;
  }

 private:
  ncclComm_t comm_;
  int rank_;
  int world_size_;
};

class NcclKernel : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit NcclKernel(const OpKernelInfo& info);

 protected:
  NcclContext* nccl_ = nullptr;
};

/*
 * Defines new version of Nccl classes that independent with training::DistributedContext
 * only rely on MPI
 */
class AllReduce final : public NcclKernel {
 public:
  explicit AllReduce(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

class AllGather final : public NcclKernel {
 public:
  explicit AllGather(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t group_size_ = -1;
  int64_t axis_ = -1;
  const CUDAExecutionProvider* cuda_ep_;
};

class AllToAll final : public NcclKernel {
 public:
  explicit AllToAll(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t group_size_ = -1;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
