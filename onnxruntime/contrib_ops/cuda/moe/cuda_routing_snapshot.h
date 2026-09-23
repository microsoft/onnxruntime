// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/kernel_pilot.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

class CudaRoutingSnapshot {
 public:
  explicit CudaRoutingSnapshot(AllocatorPtr pinned_allocator)
      : pinned_allocator_(std::move(pinned_allocator)) {}

  ~CudaRoutingSnapshot() {
    ORT_IGNORE_RETURN_VALUE(WaitForCopy());
    if (copy_ready_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(copy_ready_)));
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaRoutingSnapshot);

  Status BeginInvocation(KernelPilot::MoeExpertSelection& usage, size_t expert_count) {
    // A failed invocation may have returned after enqueueing a copy but before consuming it.
    ORT_RETURN_IF_ERROR(WaitForCopy());
    ORT_RETURN_IF_ERROR(usage.BeginInvocation(expert_count));
    usage_ = &usage;
    return Status::OK();
  }

  Status Capture(const int* expert_ids, size_t count, cudaStream_t stream) {
    ORT_RETURN_IF_NOT(usage_, "Kernel usage collection was not initialized.");
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
  static void CaptureRouting(void* snapshot, const int* expert_ids, size_t count, cudaStream_t stream) {
    ORT_THROW_IF_ERROR(static_cast<CudaRoutingSnapshot*>(snapshot)->Capture(expert_ids, count, stream));
  }

  Status Consume() {
    ORT_RETURN_IF_NOT(copy_pending_, "No MoE routing snapshot is available to consume.");
    ORT_RETURN_IF_ERROR(WaitForCopy());
    return usage_->Collect(gsl::make_span(host_ids_.get(), captured_count_));
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
  KernelPilot::MoeExpertSelection* usage_{nullptr};
  IAllocatorUniquePtr<int> host_ids_;
  size_t capacity_{0};
  size_t captured_count_{0};
  cudaEvent_t copy_ready_{nullptr};
  cudaStream_t copy_stream_{nullptr};
  bool copy_pending_{false};
  bool copy_recorded_{false};
};

}  // namespace onnxruntime::contrib::cuda
