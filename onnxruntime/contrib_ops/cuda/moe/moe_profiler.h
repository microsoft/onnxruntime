// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <type_traits>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
std::string CudaMoeJsonArray(gsl::span<const T> values) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ",";
    }
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isfinite(values[i])) {
        stream << std::setprecision(std::numeric_limits<T>::max_digits10) << values[i];
      } else {
        stream << "null";
      }
    } else {
      stream << values[i];
    }
  }
  stream << "]";
  return stream.str();
}

class CudaMoeRoutingRecord final : public DeferredRunInstrumentationRecord {
 public:
  CudaMoeRoutingRecord(const RunInstrumentationContext& instrumentation,
                       std::string node_name,
                       NodeIndex node_index,
                       IAllocatorUniquePtr<int> expert_ids,
                       IAllocatorUniquePtr<float> router_weights,
                       size_t element_count,
                       int64_t num_rows,
                       int64_t top_k,
                       int device_id,
                       TimePoint start_time)
      : instrumentation_(instrumentation),
        node_name_(std::move(node_name)),
        node_index_(node_index),
        expert_ids_(std::move(expert_ids)),
        router_weights_(std::move(router_weights)),
        element_count_(element_count),
        num_rows_(num_rows),
        top_k_(top_k),
        device_id_(device_id),
        start_time_(start_time) {}

  ~CudaMoeRoutingRecord() override {
    DestroyEvents();
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaMoeRoutingRecord);

  Status Start(cudaStream_t stream) {
    CUDA_RETURN_IF_ERROR(cudaEventCreate(&start_event_));
    CUDA_RETURN_IF_ERROR(cudaEventCreate(&completion_event_));
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&snapshot_event_, cudaEventDisableTiming));
    CUDA_RETURN_IF_ERROR(cudaEventRecord(start_event_, stream));
    return Status::OK();
  }

  Status CaptureTile(const int* device_expert_ids,
                     const float* device_router_weights,
                     size_t destination_offset,
                     size_t element_count,
                     bool is_last_tile,
                     cudaStream_t stream) {
    ORT_RETURN_IF(destination_offset + element_count > element_count_,
                  "CUDA MoE routing snapshot exceeds its pinned host buffers.");
    if (is_last_tile) {
      CUDA_RETURN_IF_ERROR(cudaEventRecord(completion_event_, stream));
      completion_recorded_ = true;
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(expert_ids_.get() + destination_offset,
                                         device_expert_ids,
                                         element_count * sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(router_weights_.get() + destination_offset,
                                         device_router_weights,
                                         element_count * sizeof(float),
                                         cudaMemcpyDeviceToHost,
                                         stream));
    CUDA_RETURN_IF_ERROR(cudaEventRecord(snapshot_event_, stream));
    snapshot_recorded_ = true;
    return Status::OK();
  }

  std::string Emit() override {
    const Status status = EmitInternal();
    return status.IsOK() ? std::string{} : status.ErrorMessage();
  }

 private:
  Status EmitInternal() {
    ORT_RETURN_IF_NOT(completion_recorded_,
                      "CUDA MoE routing snapshot did not record a completion event.");

    int previous_device = 0;
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device));
    if (previous_device != device_id_) {
      CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id_));
    }
    auto restore_device = gsl::finally([previous_device, this]() {
      DestroyEvents();
      if (previous_device != device_id_) {
        ORT_IGNORE_RETURN_VALUE(cudaSetDevice(previous_device));
      }
    });

    ORT_RETURN_IF_NOT(snapshot_recorded_,
                      "CUDA MoE routing snapshot did not record a copy-completion event.");
    cudaError_t query_result = cudaEventQuery(snapshot_event_);
    if (query_result == cudaErrorNotReady) {
      CUDA_RETURN_IF_ERROR(cudaEventSynchronize(snapshot_event_));
      query_result = cudaSuccess;
    }
    CUDA_RETURN_IF_ERROR(query_result);

    float elapsed_ms = 0.0f;
    CUDA_RETURN_IF_ERROR(cudaEventElapsedTime(&elapsed_ms, start_event_, completion_event_));
    const auto elapsed_duration =
        std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<float, std::milli>(elapsed_ms));
    const TimePoint completion_time = start_time_ + elapsed_duration;
    const int64_t completion_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(completion_time.time_since_epoch()).count() -
        static_cast<int64_t>(instrumentation_.ProfilerStartTimeNs());

    instrumentation_.RecordMoeRoutingEvent(
        start_time_, completion_time, node_name_, node_index_,
        CudaMoeJsonArray(gsl::make_span(static_cast<const int*>(expert_ids_.get()), element_count_)),
        CudaMoeJsonArray(gsl::make_span(static_cast<const float*>(router_weights_.get()), element_count_)),
        num_rows_, top_k_, device_id_, completion_ns,
        "cuda_event_elapsed_from_host_enqueue");
    return Status::OK();
  }

  void DestroyEvents() noexcept {
    if (start_event_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(cudaEventDestroy(start_event_));
      start_event_ = nullptr;
    }
    if (completion_event_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(cudaEventDestroy(completion_event_));
      completion_event_ = nullptr;
    }
    if (snapshot_event_ != nullptr) {
      if (snapshot_recorded_) {
        ORT_IGNORE_RETURN_VALUE(cudaEventSynchronize(snapshot_event_));
      }
      ORT_IGNORE_RETURN_VALUE(cudaEventDestroy(snapshot_event_));
      snapshot_event_ = nullptr;
    }
  }

  const RunInstrumentationContext& instrumentation_;
  std::string node_name_;
  NodeIndex node_index_;
  IAllocatorUniquePtr<int> expert_ids_;
  IAllocatorUniquePtr<float> router_weights_;
  size_t element_count_;
  int64_t num_rows_;
  int64_t top_k_;
  int device_id_;
  TimePoint start_time_;
  cudaEvent_t start_event_{};
  cudaEvent_t completion_event_{};
  cudaEvent_t snapshot_event_{};
  bool completion_recorded_{false};
  bool snapshot_recorded_{false};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
