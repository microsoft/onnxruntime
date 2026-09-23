// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

class CudaMoeExpertCounter {
 public:
  explicit CudaMoeExpertCounter(AllocatorPtr pinned_allocator)
      : pinned_allocator_(std::move(pinned_allocator)) {}

  ~CudaMoeExpertCounter() {
    ORT_IGNORE_RETURN_VALUE(WaitForCopy());
    if (copy_ready_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(copy_ready_)));
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaMoeExpertCounter);

  Status BeginInvocation(size_t expert_count) {
    ORT_RETURN_IF(expert_count == 0, "MoE expert collection requires a positive expert count.");
    // A failed invocation may have returned after enqueueing a copy but before consuming it.
    ORT_RETURN_IF_ERROR(WaitForCopy());
    used_.assign(expert_count, 0);
    selected_experts_.clear();
    selected_experts_.reserve(expert_count);
    return Status::OK();
  }

  Status Capture(const int* expert_ids, size_t count, cudaStream_t stream) {
    ORT_RETURN_IF(used_.empty(), "MoE expert collection was not initialized for this invocation.");
    ORT_RETURN_IF(copy_pending_, "The previous MoE routing snapshot has not been consumed.");
    cudaStreamCaptureStatus capture_status;
    CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture_status));
    ORT_RETURN_IF(capture_status != cudaStreamCaptureStatusNone,
                  "MoE expert counting is not supported during CUDA graph capture.");
    if (count > capacity_) {
      auto buffer = IAllocator::MakeUniquePtr<int>(pinned_allocator_, count);
      ORT_RETURN_IF_NOT(buffer, "Failed to allocate pinned host memory for MoE expert counting.");
      host_ids_ = std::move(buffer);
      capacity_ = count;
    }
    if (copy_ready_ == nullptr) {
      CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&copy_ready_, cudaEventDisableTiming));
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_ids_.get(), expert_ids, SafeInt<size_t>(count) * sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
    copy_pending_ = true;
    copy_stream_ = stream;
    captured_count_ = count;
    CUDA_RETURN_IF_ERROR(cudaEventRecord(copy_ready_, stream));
    copy_recorded_ = true;
    return Status::OK();
  }

  // The fused-routing runner invokes this on the calling CPU thread, before launching expert GEMMs.
  static void CaptureRouting(void* counter, const int* expert_ids, size_t count, cudaStream_t stream) {
    ORT_THROW_IF_ERROR(static_cast<CudaMoeExpertCounter*>(counter)->Capture(expert_ids, count, stream));
  }

  Status Consume() {
    ORT_RETURN_IF_NOT(copy_pending_, "No MoE routing snapshot is available to consume.");
    ORT_RETURN_IF_ERROR(WaitForCopy());
    for (size_t row = 0; row < captured_count_; ++row) {
      const int expert = host_ids_.get()[row];
      ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= used_.size(),
                    "MoE counter expert index out of range: ", expert);
      if (!used_[expert]) {
        used_[expert] = 1;
        selected_experts_.push_back(expert);
      }
    }
    return Status::OK();
  }

  Status GetSelectedExperts(gsl::span<const int>& expert_ids) const {
    ORT_RETURN_IF(used_.empty(), "MoE expert collection was not initialized for this invocation.");
    ORT_RETURN_IF(copy_pending_, "The MoE routing snapshot must be consumed before reading selected experts.");
    expert_ids = selected_experts_;
    return Status::OK();
  }

 private:
  Status WaitForCopy() {
    if (copy_pending_) {
      if (copy_recorded_) {
        CUDA_RETURN_IF_ERROR(cudaEventSynchronize(copy_ready_));
      } else {
        // Only error recovery uses a stream wait: recording the copy event failed after enqueueing the DMA.
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(copy_stream_));
      }
      copy_pending_ = false;
      copy_recorded_ = false;
    }
    return Status::OK();
  }

  AllocatorPtr pinned_allocator_;
  InlinedVector<uint8_t> used_;
  InlinedVector<int> selected_experts_;
  IAllocatorUniquePtr<int> host_ids_;
  size_t capacity_{0};
  size_t captured_count_{0};
  cudaEvent_t copy_ready_{nullptr};
  cudaStream_t copy_stream_{nullptr};
  bool copy_pending_{false};
  bool copy_recorded_{false};
};

}  // namespace onnxruntime::contrib::cuda
