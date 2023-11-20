// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#if defined(ORT_USE_NCCL)
#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define NCCL_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(NCCL_CALL(expr))

static ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type) {
  if (type == DataTypeImpl::GetType<uint8_t>()) {
    return ncclUint8;
  } else if (type == DataTypeImpl::GetType<bool>()) {
    // CUDA bool is 8-bit large.
    return ncclUint8;
  } else if (type == DataTypeImpl::GetType<int8_t>()) {
    return ncclInt8;
  } else if (type == DataTypeImpl::GetType<int32_t>()) {
    return ncclInt32;
  } else if (type == DataTypeImpl::GetType<int64_t>()) {
    return ncclInt64;
  } else if (type == DataTypeImpl::GetType<MLFloat16>()) {
    return ncclFloat16;
  } else if (type == DataTypeImpl::GetType<float>()) {
    return ncclFloat32;
  } else if (type == DataTypeImpl::GetType<double>()) {
    return ncclFloat64;
  } else {
    ORT_THROW("Tensor type not supported in NCCL.");
  }
}

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

  ncclComm_t Comm() const {
    return nccl_->Comm();
  }

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

Status FuncAllReduce(
    ncclComm_t comm,
    cudaStream_t stream,
    const Tensor* input,
    Tensor* output);

void FuncAllGather(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const Tensor* input,
    const int64_t group_size,
    const int64_t axis,
    Tensor* output);

std::unique_ptr<Tensor> FuncAllGather(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const Tensor* input,
    const int64_t group_size,
    const int64_t axis);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
