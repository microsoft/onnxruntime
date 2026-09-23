// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/safeint.h"
#include "core/framework/moe_expert_usage.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

class CudaMoeExpertCounter {
 public:
  CudaMoeExpertCounter() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaMoeExpertCounter);

  void BeginInvocation(MoeExpertUsage& usage, size_t expert_count) {
    usage_ = &usage;
    used_.assign(expert_count, 0);
    selected_experts_.clear();
    selected_experts_.reserve(expert_count);
  }

  Status Capture(const int* expert_ids, size_t count, cudaStream_t stream) {
    ORT_RETURN_IF_NOT(usage_, "MoE expert collection was not initialized for this invocation.");
    cudaStreamCaptureStatus capture_status;
    CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
    ORT_RETURN_IF(capture_status != cudaStreamCaptureStatusNone,
                  "MoE expert counting is not supported during CUDA graph capture.");
    while (host_ids_.size() < count) {
      host_ids_.push_back(0);
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_ids_.data(), expert_ids, SafeInt<size_t>(count) * sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
    // Counting is opt-in. Finish reading each tile before its routing scratch buffer is reused.
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    for (size_t row = 0; row < count; ++row) {
      const int expert = host_ids_[row];
      ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= used_.size(),
                    "MoE counter expert index out of range: ", expert);
      if (!used_[expert]) {
        used_[expert] = 1;
        selected_experts_.push_back(expert);
      }
    }
    return Status::OK();
  }

  Status Record() const {
    ORT_RETURN_IF_NOT(usage_, "MoE expert collection was not initialized for this invocation.");
    return usage_->RecordUsage(selected_experts_);
  }

 private:
  MoeExpertUsage* usage_{nullptr};
  InlinedVector<uint8_t> used_;
  InlinedVector<int> selected_experts_;
  InlinedVector<int> host_ids_;
};

}  // namespace onnxruntime::contrib::cuda
