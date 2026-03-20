// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_graph_plugin.h"

#include <cuda_runtime_api.h>
#include <string>
#include <stdexcept>

namespace onnxruntime {
namespace cuda_plugin {

// ---------------------------------------------------------------------------
// CudaGraphSet
// ---------------------------------------------------------------------------

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

void CudaGraphSet::Clear() {
  for (auto& [id, graph_exec] : cuda_graphs_) {
    (void)cudaGraphExecDestroy(graph_exec);
  }
  cuda_graphs_.clear();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t id) const {
  return cuda_graphs_.find(id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec) {
  if (Contains(id)) {
    throw std::runtime_error(
        "CudaGraphSet::Put: annotation id " + std::to_string(id) +
        " already exists. Use a different annotation id.");
  }
  cuda_graphs_.emplace(id, graph_exec);
}

cudaGraphExec_t CudaGraphSet::Get(CudaGraphAnnotation_t id) const {
  auto it = cuda_graphs_.find(id);
  if (it == cuda_graphs_.end()) {
    throw std::runtime_error(
        "CudaGraphSet::Get: no graph found for annotation id " + std::to_string(id));
  }
  return it->second;
}

// ---------------------------------------------------------------------------
// CUDAGraphManager
// ---------------------------------------------------------------------------

CUDAGraphManager::CUDAGraphManager(cudaStream_t stream) : stream_(stream) {}

CUDAGraphManager::~CUDAGraphManager() {
  Reset();
}

void CUDAGraphManager::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CUDAGraphManager::CaptureBegin(CudaGraphAnnotation_t annotation_id) {
  if (!IsGraphCaptureAllowedOnRun(annotation_id)) {
    throw std::runtime_error("CUDAGraphManager::CaptureBegin: capture not allowed for annotation " +
                             std::to_string(annotation_id));
  }

  if (cuda_graph_set_.Contains(annotation_id)) {
    throw std::runtime_error(
        "CUDAGraphManager::CaptureBegin: annotation id " + std::to_string(annotation_id) +
        " already captured. Use a different annotation id.");
  }

  auto err = cudaStreamSynchronize(stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(err));
  }

  // cudaStreamCaptureModeGlobal: single-thread capture (future: ThreadLocal for multi-stream)
  err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamBeginCapture failed: ") + cudaGetErrorString(err));
  }
}

void CUDAGraphManager::CaptureEnd(CudaGraphAnnotation_t annotation_id) {
  cudaGraph_t graph = nullptr;
  auto err = cudaStreamEndCapture(stream_, &graph);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamEndCapture failed: ") + cudaGetErrorString(err));
  }
  if (graph == nullptr) {
    throw std::runtime_error("CUDAGraphManager::CaptureEnd: captured graph is NULL");
  }

  cudaGraphExec_t graph_exec = nullptr;
  err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  (void)cudaGraphDestroy(graph);

  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(err));
  }

  cuda_graph_set_.Put(annotation_id, graph_exec);
}

OrtStatus* CUDAGraphManager::Replay(CudaGraphAnnotation_t annotation_id, bool sync) {
  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(annotation_id);

  auto err = cudaGraphLaunch(graph_exec, stream_);
  if (err != cudaSuccess) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL,
        (std::string("cudaGraphLaunch failed: ") + cudaGetErrorString(err)).c_str());
  }

  if (sync) {
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("cudaStreamSynchronize after graph replay failed: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return nullptr;
}

bool CUDAGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t annotation_id) const {
  return annotation_id != kCudaGraphAnnotationSkip;
}

bool CUDAGraphManager::IsGraphCaptured(CudaGraphAnnotation_t annotation_id) const {
  return cuda_graph_set_.Contains(annotation_id);
}

void CUDAGraphManager::Reset() {
  cuda_graph_set_.Clear();
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
