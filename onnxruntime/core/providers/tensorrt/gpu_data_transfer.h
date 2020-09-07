// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

class GPUDataTransfer : public Provider_IDataTransfer {
 public:
  GPUDataTransfer();
  ~GPUDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Suppress MSVC warning about not fully overriding
  using Provider_IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Provider_Tensor& src, Provider_Tensor& dst, int exec_queue_id) const override;

  cudaStream_t GetStream(int queue_id) const {
    ORT_ENFORCE(queue_id >= 0 && queue_id < kTotalCudaStreams);
    return streams_[queue_id];
  }

 private:
  cudaStream_t streams_[kTotalCudaStreams];
};

}  // namespace onnxruntime
