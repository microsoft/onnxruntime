// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/hip_common.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include <rccl.h>

namespace onnxruntime {
namespace hip {

#define NCCL_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(RCCL_CALL(expr) ? common::Status::OK() : common::Status(common::ONNXRUNTIME, common::FAIL))

class NcclContext final {
 public:
  NcclContext();
  ~NcclContext();

  ncclComm_t Comm(training::WorkerGroupType group_type);

  int Rank(training::WorkerGroupType group_type) const {
    return training::DistributedRunContext::RankInGroup(group_type);
  }

  int Size(training::WorkerGroupType group_type) const {
    return training::DistributedRunContext::GroupSize(group_type);
  }

 private:
  ncclComm_t data_group_comm_;
  ncclComm_t horizontal_group_comm_;
};

// -----------------------------------------------------------------------
// Base class for NCCL kernels
// -----------------------------------------------------------------------
class NcclKernel : public HipKernel {
 public:
  explicit NcclKernel(const OpKernelInfo& info);

 protected:
  NcclContext* nccl_ = nullptr;
  training::WorkerGroupType group_type_;
};

ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type);

}  // namespace hip
}  // namespace onnxruntime
