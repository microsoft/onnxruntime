// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_pch.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"
#include "shared_inc/cuda_utils.h"
#include <deque>

#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/memcpy.h"
#include "cuda_fence.h"
#include "cuda_allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/contrib_kernels.h"

namespace onnxruntime {

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer();

  virtual bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_Device) const override;

  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const override;

 private:
  cudaStream_t streams_[kTotalCudaStreams];
};

}  // namespace onnxruntime
