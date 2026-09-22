// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

class CudaMoeExpertCounter {
 public:
  explicit CudaMoeExpertCounter(const OpKernelContext* context) : context_(context) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaMoeExpertCounter);

  Status Capture(const int* expert_ids, size_t count, cudaStream_t stream) {
    if (!context_->HasMoeExpertState()) {
      return Status::OK();
    }
    cudaStreamCaptureStatus capture_status;
    CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
    ORT_RETURN_IF(capture_status != cudaStreamCaptureStatusNone,
                  "MoE expert counting is not supported during CUDA graph capture.");
    InlinedVector<int> host_ids(count);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_ids.data(), expert_ids, SafeInt<size_t>(count) * sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
    // Counting is opt-in. Finish reading each tile before its routing scratch buffer is reused.
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    used_.insert(host_ids.begin(), host_ids.end());
    return Status::OK();
  }

  Status Record() const {
    if (!context_->HasMoeExpertState()) {
      return Status::OK();
    }
    const InlinedVector<int> ids(used_.begin(), used_.end());
    return context_->RecordMoeExpertUsage(ids);
  }

 private:
  const OpKernelContext* context_;
  InlinedHashSet<int> used_;
};

}  // namespace onnxruntime::contrib::cuda
