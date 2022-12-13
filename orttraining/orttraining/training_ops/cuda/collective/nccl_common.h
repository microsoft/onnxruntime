// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "orttraining/core/framework/distributed_run_context.h"

#if defined(ORT_USE_NCCL)
#include <nccl.h>
#endif

namespace onnxruntime {
namespace cuda {

#if defined(ORT_USE_NCCL)
#define NCCL_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(NCCL_CALL(expr))
#endif
class NcclContext final {
 public:
  NcclContext();
  ~NcclContext();

  ncclComm_t Comm(training::WorkerGroupType group_type);

  int Rank(training::WorkerGroupType group_type) const {
    //return training::DistributedRunContext::RankInGroup(group_type);
    ORT_UNUSED_PARAMETER(group_type);
    return rank_;
  }

  int Size(training::WorkerGroupType group_type) const {
    // return training::DistributedRunContext::GroupSize(group_type);
    ORT_UNUSED_PARAMETER(group_type);
    return world_size_;
  }

 private:
  ncclComm_t global_group_comm_;
  ncclComm_t data_group_comm_;
  ncclComm_t node_local_comm_;
  ncclComm_t cross_node_comm_;
  ncclComm_t horizontal_group_comm_;
  int world_size_;
  int rank_;
};

// -----------------------------------------------------------------------
// Base class for NCCL kernels
// -----------------------------------------------------------------------
class NcclKernel : public CudaKernel {
 public:
  explicit NcclKernel(const OpKernelInfo& info);

 protected:
  NcclContext* nccl_ = nullptr;
  training::WorkerGroupType group_type_;
};

ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type);

// -----------------------------------------------------------------------
// Defines a new version of nccl classes
// that independent with training::DistributedRunContext, only rely on MPI
// -----------------------------------------------------------------------
class NcclContextV2 final {
 public:
  NcclContextV2();
  ~NcclContextV2();

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

// -----------------------------------------------------------------------
// Base class for NCCL kernels version 2
// -----------------------------------------------------------------------
class NcclKernelV2 : public CudaKernel {
 public:
  explicit NcclKernelV2(const OpKernelInfo& info);

 protected:
  NcclContextV2* nccl_ = nullptr;
};

}  // namespace cuda
}  // namespace onnxruntime
