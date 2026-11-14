// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_graph.h"

#include "core/providers/cuda/cuda_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gsl/gsl>

namespace onnxruntime {

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

void CudaGraphSet::Clear() {
  for (auto& it : cuda_graphs_) {
    CUDA_CALL_THROW(cudaGraphExecDestroy(it.second));
  }
  cuda_graphs_.clear();
}

Status CudaGraphSet::Remove(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  auto it = cuda_graphs_.find(cuda_graph_annotation_id);
  if (it != cuda_graphs_.end()) {
    CUDA_RETURN_IF_ERROR(cudaGraphExecDestroy(it->second));
    cuda_graphs_.erase(it);
  }
  return Status::OK();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graphs_.find(cuda_graph_annotation_id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec) {
  ORT_ENFORCE(!Contains(cuda_graph_annotation_id));
  cuda_graphs_.emplace(cuda_graph_annotation_id, graph_exec);
}

cudaGraphExec_t CudaGraphSet::Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  ORT_ENFORCE(Contains(cuda_graph_annotation_id));
  return cuda_graphs_.at(cuda_graph_annotation_id);
}

CUDAGraphManager::CUDAGraphManager(cudaStream_t stream) : stream_(stream) {
}

void CUDAGraphManager::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CUDAGraphManager::CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  ORT_ENFORCE(IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id));
  ORT_ENFORCE(cuda_graph_annotation_id_ == kCudaGraphAnnotationSkip,
              "A CUDA graph capture is already active with annotation id ", cuda_graph_annotation_id_);

  ORT_ENFORCE(!cuda_graph_set_.Contains(cuda_graph_annotation_id),
              "Trying to capture a graph with annotation id ", cuda_graph_annotation_id,
              " that already used. Please use a different annotation id.");

  CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
  // Begin and end each capture on the same caller thread. Serialized runs may
  // use different caller threads while sharing this stream and graph manager.
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  cuda_graph_annotation_id_ = cuda_graph_annotation_id;
}

void CUDAGraphManager::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  ORT_ENFORCE(cuda_graph_annotation_id_ != kCudaGraphAnnotationSkip,
              "No ORT-owned CUDA graph capture is active");
  ORT_ENFORCE(cuda_graph_annotation_id == cuda_graph_annotation_id_,
              "CUDA graph capture annotation id mismatch: expected ", cuda_graph_annotation_id_,
              " but received ", cuda_graph_annotation_id);

  cudaGraph_t graph = NULL;
  const auto capture_status = cudaStreamEndCapture(stream_, &graph);
  cuda_graph_annotation_id_ = kCudaGraphAnnotationSkip;
  CUDA_CALL_THROW(capture_status);
  if (graph == NULL) {
    ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
  }

  cudaGraphExec_t graph_exec = NULL;
  const auto release_graph_exec_on_error = gsl::finally([&]() {
    if (graph_exec != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaGraphExecDestroy(graph_exec)));
    }
  });
  const auto instantiate_status = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
  const auto destroy_status = cudaGraphDestroy(graph);
  if (instantiate_status != cudaSuccess && destroy_status != cudaSuccess) {
    // Cleanup failure must not mask the instantiation error.
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(destroy_status));
  }
  CUDA_CALL_THROW(instantiate_status);
  CUDA_CALL_THROW(destroy_status);

  cuda_graph_set_.Put(cuda_graph_annotation_id, graph_exec);
  graph_exec = nullptr;
}

void CUDAGraphManager::DiscardCapture() noexcept {
  cudaGraph_t graph = NULL;
  // A capture invalidated by the failed run reports that here and yields no graph. The stream leaves capture mode
  // either way.
  const cudaError_t end_status = cudaStreamEndCapture(stream_, &graph);
  cuda_graph_annotation_id_ = kCudaGraphAnnotationSkip;
  if (graph != NULL) {
    static_cast<void>(cudaGraphDestroy(graph));
  }
  if (end_status != cudaSuccess) {
    // Clear the recorded error so it is not attributed to later CUDA calls on this thread.
    static_cast<void>(cudaGetLastError());
  }
}

Status CUDAGraphManager::Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync_status_flag) {
  // The graph-enabled CUDA EP serializes execution and release through the session.
  LOGS_DEFAULT(INFO) << "Replaying CUDA graph on stream " << stream_ << " with cuda_graph_annotation_id "
                     << cuda_graph_annotation_id;

  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(cuda_graph_annotation_id);
  CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph_exec, stream_));

  if (sync_status_flag) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
  }
  return Status::OK();
}

Status CUDAGraphManager::Release(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  if (cuda_graph_set_.Contains(cuda_graph_annotation_id)) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
    ORT_RETURN_IF_ERROR(cuda_graph_set_.Remove(cuda_graph_annotation_id));
  }
  return Status::OK();
}

CudaGraphAnnotation_t CUDAGraphManager::CapturingGraphId() const {
  return cuda_graph_annotation_id_;
}

bool CUDAGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_annotation_id != kCudaGraphAnnotationSkip;
}

bool CUDAGraphManager::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_set_.Contains(cuda_graph_annotation_id);
}

void CUDAGraphManager::Reset() {
  cuda_graph_set_.Clear();
}

CUDAGraphManager::~CUDAGraphManager() {
  Reset();
}

}  // namespace onnxruntime
