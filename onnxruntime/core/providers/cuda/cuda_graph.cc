// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_graph.h"

#include "core/providers/cuda/cuda_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace onnxruntime {

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

bool CudaGraphSet::IsEmpty() const {
  return cuda_graphs_.empty();
}

void CudaGraphSet::Clear() {
  for (auto& it : cuda_graphs_) {
    CUDA_CALL_THROW(cudaGraphExecDestroy(it.second));
  }
  cuda_graphs_.clear();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graphs_.find(cuda_graph_annotation_id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec) {
  if (Contains(cuda_graph_annotation_id)) {
    // This should not happen
    throw std::runtime_error("CudaGraphSet::Put: cuda_graph_annotation_id already exists");
  }
  cuda_graphs_.emplace(cuda_graph_annotation_id, graph_exec);
}

bool CudaGraphSet::Get(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t& graph_exec) {
  if (!Contains(cuda_graph_annotation_id)) {
    return false;
  }
  graph_exec = cuda_graphs_.at(cuda_graph_annotation_id);
  return true;
}

CUDAGraphManager::CUDAGraphManager(cudaStream_t stream) : stream_(stream) {
}

void CUDAGraphManager::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CUDAGraphManager::SetGraphAnnotationId(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  cuda_graph_annotation_id_ = cuda_graph_annotation_id;
}

void CUDAGraphManager::CaptureBegin() {
  if (!IsGraphCaptureAllowedOnRun()) {
    LOGS_DEFAULT(INFO) << "Skipping graph capture for cuda_graph_annotation_id " << cuda_graph_annotation_id_;
    return;
  }

  ORT_ENFORCE(!cuda_graph_set_.Contains(cuda_graph_annotation_id_),
              "This cuda graph has already captured a graph. "
              "Create a new instance to capture a new graph.");

  CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
  // For now cuda graph can only work with a single thread. In the future, we
  // will support multiple threads. For multiple threads with multiple graphs
  // and streams, `cudaStreamCaptureModeGlobal` needs to be changed to
  // `cudaStreamCaptureModeThreadLocal`
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
}

void CUDAGraphManager::CaptureEnd() {
  if (!IsGraphCaptureAllowedOnRun()) {
    LOGS_DEFAULT(INFO) << "Skipping graph capture for cuda_graph_annotation_id " << cuda_graph_annotation_id_;
    return;
  }

  CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph_));
  if (graph_ == NULL) {
    ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
  }
  has_graph_ = true;

  cudaGraphExec_t graph_exec = NULL;
  CUDA_CALL_THROW(cudaGraphInstantiate(&graph_exec, graph_, NULL, NULL, 0));
  CUDA_CALL_THROW(cudaGraphDestroy(graph_));
  has_graph_ = false;

  cuda_graph_set_.Put(cuda_graph_annotation_id_, graph_exec);
  has_graph_exec_ = true;
}

Status CUDAGraphManager::Replay() {
  if (!IsGraphCaptureAllowedOnRun()) {
    LOGS_DEFAULT(INFO) << "Skipping graph replay for cuda_graph_annotation_id " << cuda_graph_annotation_id_;
    return Status::OK();
  }
  // Although this function is not thread safe, the lock is not needed here because
  // CUDA EP maintains a separate cuda graph per thread
  LOGS_DEFAULT(INFO) << "Replaying CUDA graph on stream " << stream_ << " with cuda_graph_annotation_id " << cuda_graph_annotation_id_;

  cudaGraphExec_t graph_exec = NULL;
  if (!cuda_graph_set_.Get(cuda_graph_annotation_id_, graph_exec)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "CUDAGraph::Replay: cuda_graph_set_ does not contain the cuda_graph_annotation_id");
  }
  CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph_exec, stream_));

  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
  return Status::OK();
}

bool CUDAGraphManager::IsGraphCaptureAllowedOnRun() const {
  return cuda_graph_annotation_id_ != kCudaGraphAnnotationSkip;
}

bool CUDAGraphManager::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  if (!IsGraphCaptureAllowedOnRun()) {
    return false;
  }
  return cuda_graph_set_.Contains(cuda_graph_annotation_id);
}

void CUDAGraphManager::Reset() {
  if (has_graph_) {
    CUDA_CALL_THROW(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (!cuda_graph_set_.IsEmpty()) {
    cuda_graph_set_.Clear();
  }
}

CUDAGraphManager::~CUDAGraphManager() {
  Reset();
}

}  // namespace onnxruntime
