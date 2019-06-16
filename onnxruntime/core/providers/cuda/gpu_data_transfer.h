// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_pch.h"
#include "core/framework/data_transfer.h"
#include "cuda_common.h"

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
