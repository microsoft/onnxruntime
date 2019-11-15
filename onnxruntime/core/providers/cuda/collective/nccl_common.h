// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include <nccl.h>

namespace onnxruntime {
namespace cuda {

#define NCCL_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(NCCL_CALL(expr) ? common::Status::OK() : common::Status(common::ONNXRUNTIME, common::FAIL))

class NcclContext final {
 public:
  NcclContext();
  ~NcclContext();

  ncclComm_t Comm() const { return comm_; }

  int Rank() const { return rank_; }
  int Size() const { return size_; }
  int LocalRank() const { return local_rank_; }
  int LocalSize() const { return local_size_; }

 private:
  ncclComm_t comm_;
  int rank_ = 0;
  int size_ = 1;
  int local_rank_ = 0;
  int local_size_ = 1;
};

// -----------------------------------------------------------------------
// Base class for NCCL kernels
// -----------------------------------------------------------------------
class NcclKernel : public CudaKernel {
 public:
  explicit NcclKernel(const OpKernelInfo& info);

 protected:
  NcclContext* nccl_ = nullptr;
};

ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type);

}  // namespace cuda
}  // namespace onnxruntime
