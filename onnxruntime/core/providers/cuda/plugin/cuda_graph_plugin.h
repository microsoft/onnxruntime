// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin-compatible CUDA graph manager for capture/replay lifecycle.
// Adapted from core/providers/cuda/cuda_graph.h — removes dependencies
// on internal EP types (CUDAExecutionProvider, CudaStream).

#pragma once

#include "cuda_plugin_utils.h"

#include <unordered_map>
#include <mutex>

namespace onnxruntime {
namespace cuda_plugin {

using CudaGraphAnnotation_t = int;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

/// Stores instantiated CUDA graph executables keyed by annotation ID.
struct CudaGraphSet {
  CudaGraphSet() = default;
  ~CudaGraphSet();

  void Clear();
  bool Contains(CudaGraphAnnotation_t id) const;
  void Put(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec);
  cudaGraphExec_t Get(CudaGraphAnnotation_t id) const;

 private:
  std::unordered_map<CudaGraphAnnotation_t, cudaGraphExec_t> cuda_graphs_;
};

/// Manages CUDA graph capture/instantiation/replay for the plugin EP.
/// Each instance is associated with a single cudaStream_t.
struct CUDAGraphManager {
  CUDAGraphManager() = default;
  explicit CUDAGraphManager(cudaStream_t stream);
  ~CUDAGraphManager();

  void SetStream(cudaStream_t stream);

  /// Begin capturing CUDA work on the associated stream.
  void CaptureBegin(CudaGraphAnnotation_t annotation_id);

  /// End capture, instantiate the graph, and store it.
  void CaptureEnd(CudaGraphAnnotation_t annotation_id);

  /// Launch a previously captured graph.
  OrtStatus* Replay(CudaGraphAnnotation_t annotation_id, bool sync = true);

  /// Destroy all captured graphs.
  void Reset();

  /// Whether capture is allowed for the given annotation (i.e., not the skip sentinel).
  bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t annotation_id) const;

  /// Whether a graph has already been captured for the given annotation.
  bool IsGraphCaptured(CudaGraphAnnotation_t annotation_id) const;

 private:
  CudaGraphSet cuda_graph_set_;
  cudaStream_t stream_ = nullptr;  // Does not own the stream
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
