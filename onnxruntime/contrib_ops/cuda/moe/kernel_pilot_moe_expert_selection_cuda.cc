// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.h"

#include <utility>

#include "core/common/safeint.h"

namespace onnxruntime::contrib::cuda {

KernelPilotMoeExpertSelectionCuda::KernelPilotMoeExpertSelectionCuda(AllocatorPtr pinned_allocator)
    : pinned_allocator_(std::move(pinned_allocator)) {}

KernelPilotMoeExpertSelectionCuda::~KernelPilotMoeExpertSelectionCuda() {
  ORT_IGNORE_RETURN_VALUE(WaitForCopy());
  if (copy_ready_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(copy_ready_)));
  }
}

Status KernelPilotMoeExpertSelectionCuda::BeginInvocation(IKernelPilotMoeExpertSelection& usage,
                                                          size_t expert_count) {
  usage_ = &usage;
  return BeginInvocation(expert_count);
}

Status KernelPilotMoeExpertSelectionCuda::BeginInvocation(size_t expert_count) {
  ORT_RETURN_IF_NOT(usage_, "MoE expert selection is not attached.");
  // A failed invocation may have returned after enqueueing a copy but before consuming it.
  ORT_RETURN_IF_ERROR(WaitForCopy());
  return usage_->BeginInvocation(expert_count);
}

bool KernelPilotMoeExpertSelectionCuda::IsInitialized() const noexcept {
  return usage_ != nullptr && usage_->IsInitialized();
}

size_t KernelPilotMoeExpertSelectionCuda::ExpertCount() const noexcept {
  return usage_ != nullptr ? usage_->ExpertCount() : 0;
}

Status KernelPilotMoeExpertSelectionCuda::Collect(gsl::span<const int> expert_ids) {
  ORT_RETURN_IF_NOT(usage_, "MoE expert selection is not attached.");
  return usage_->Collect(expert_ids);
}

Status KernelPilotMoeExpertSelectionCuda::GetSelectedExperts(gsl::span<const int>& expert_ids) const {
  ORT_RETURN_IF_NOT(usage_, "MoE expert selection is not attached.");
  return usage_->GetSelectedExperts(expert_ids);
}

Status KernelPilotMoeExpertSelectionCuda::Capture(const int* expert_ids, size_t count, cudaStream_t stream) {
  ORT_RETURN_IF_NOT(IsInitialized(), "Kernel usage collection was not initialized.");
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

void KernelPilotMoeExpertSelectionCuda::CaptureRouting(void* snapshot, const int* expert_ids, size_t count,
                                                       cudaStream_t stream) {
  ORT_THROW_IF_ERROR(
      static_cast<KernelPilotMoeExpertSelectionCuda*>(snapshot)->Capture(expert_ids, count, stream));
}

Status KernelPilotMoeExpertSelectionCuda::Consume() {
  ORT_RETURN_IF_NOT(copy_pending_, "No MoE routing snapshot is available to consume.");
  ORT_RETURN_IF_ERROR(WaitForCopy());
  return Collect(gsl::make_span(host_ids_.get(), captured_count_));
}

Status KernelPilotMoeExpertSelectionCuda::WaitForCopy() {
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

}  // namespace onnxruntime::contrib::cuda
