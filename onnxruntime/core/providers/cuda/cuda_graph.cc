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
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] CaptureBegin called - annotation_id=" << cuda_graph_annotation_id << ", stream=" << stream_;
  
  ORT_ENFORCE(IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id));

  ORT_ENFORCE(!cuda_graph_set_.Contains(cuda_graph_annotation_id),
              "Trying to capture a graph with annotation id ", cuda_graph_annotation_id,
              " that already used. Please use a different annotation id.");

  // Check current stream state
  cudaError_t sync_result = cudaStreamSynchronize(stream_);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] Stream sync before capture - result=" << cudaGetErrorString(sync_result);
  CUDA_CALL_THROW(sync_result);
  
  // For now cuda graph can only work with a single thread. In the future, we
  // will support multiple threads. For multiple threads with multiple graphs
  // and streams, `cudaStreamCaptureModeGlobal` needs to be changed to
  // `cudaStreamCaptureModeThreadLocal`
  cudaError_t capture_result = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] BeginCapture result=" << cudaGetErrorString(capture_result);
  CUDA_CALL_THROW(capture_result);
  
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] CaptureBegin completed successfully";
}

void CUDAGraphManager::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] CaptureEnd called - annotation_id=" << cuda_graph_annotation_id << ", stream=" << stream_;
  
  cudaGraph_t graph = NULL;
  cudaError_t end_capture_result = cudaStreamEndCapture(stream_, &graph);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] EndCapture result=" << cudaGetErrorString(end_capture_result) << ", graph=" << graph;
  CUDA_CALL_THROW(end_capture_result);
  
  if (graph == NULL) {
    LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] CRITICAL: Captured graph is NULL!";
    ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
  }

  cudaGraphExec_t graph_exec = NULL;
  cudaError_t instantiate_result = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] GraphInstantiate result=" << cudaGetErrorString(instantiate_result) << ", graph_exec=" << graph_exec;
  CUDA_CALL_THROW(instantiate_result);
  
  cudaError_t destroy_result = cudaGraphDestroy(graph);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] GraphDestroy result=" << cudaGetErrorString(destroy_result);
  CUDA_CALL_THROW(destroy_result);

  // Currently all the captured graphs will be tied to the session's lifecycle
  // TODO(wy): Addd an interface to free captured graphs
  cuda_graph_set_.Put(cuda_graph_annotation_id, graph_exec);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] CaptureEnd completed successfully - graph stored";
}

Status CUDAGraphManager::Replay(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  // Although this function is not thread safe, the lock is not needed here because
  // CUDA EP maintains a separate cuda graph per thread
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] Replay called - annotation_id=" << cuda_graph_annotation_id << ", stream=" << stream_;
  LOGS_DEFAULT(INFO) << "Replaying CUDA graph on stream " << stream_ << " with cuda_graph_annotation_id "
                     << cuda_graph_annotation_id;

  if (!cuda_graph_set_.Contains(cuda_graph_annotation_id)) {
    LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] CRITICAL: No graph found for annotation_id=" << cuda_graph_annotation_id;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No CUDA graph found for annotation ID: ", cuda_graph_annotation_id);
  }

  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(cuda_graph_annotation_id);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] Retrieved graph_exec=" << graph_exec;
  
  cudaError_t launch_result = cudaGraphLaunch(graph_exec, stream_);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] GraphLaunch result=" << cudaGetErrorString(launch_result);
  CUDA_RETURN_IF_ERROR(launch_result);

  // PERFORMANCE FIX: Remove synchronization from Replay to avoid double-sync overhead
  // The TensorRT compute function will handle synchronization when needed
  // cudaError_t sync_result = cudaStreamSynchronize(stream_);
  // LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] Stream sync after replay result=" << cudaGetErrorString(sync_result);
  // CUDA_RETURN_IF_ERROR(sync_result);
  
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] Replay completed successfully (async launch)";
  return Status::OK();
}

bool CUDAGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_annotation_id != kCudaGraphAnnotationSkip;
}

bool CUDAGraphManager::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  bool captured = cuda_graph_set_.Contains(cuda_graph_annotation_id);
  LOGS_DEFAULT(ERROR) << "[CUDA GRAPH DEBUG] IsGraphCaptured - annotation_id=" << cuda_graph_annotation_id << ", captured=" << captured;
  return captured;
}

void CUDAGraphManager::Reset() {
  cuda_graph_set_.Clear();
}

CUDAGraphManager::~CUDAGraphManager() {
  Reset();
}

}  // namespace onnxruntime
