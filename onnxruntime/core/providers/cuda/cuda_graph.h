// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/common/common.h"
#include <mutex>
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CudaGraphAnnotation_t = int;
using CudaGraphSet_t = std::unordered_map<CudaGraphAnnotation_t, cudaGraphExec_t>;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

struct CudaGraphSet {
  CudaGraphSet() {};
  ~CudaGraphSet();

  void Clear();
  bool Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
  void Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec);
  cudaGraphExec_t Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

 private:
  CudaGraphSet_t cuda_graphs_;
};

struct CUDAGraphManager {
  CUDAGraphManager() {};
  CUDAGraphManager(cudaStream_t stream);
  ~CUDAGraphManager();

  void SetStream(cudaStream_t stream);
  void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
  void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
  Status Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync_status_flag = true);

  void Reset();

  bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
  bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

  // True when `stream` is currently capturing a device graph. A null stream (the legacy
  // default stream) can never be capturing.
  static bool IsStreamCapturing(cudaStream_t stream);

  // True when a device graph capture is in progress on this manager's stream and it was started by
  // the caller rather than by any ONNX Runtime manager on that stream.
  bool IsExternalCaptureActive() const;

 private:
  // Drops this manager's claim on the stream's ORT-initiated capture, if it holds one.
  void ReleaseOrtCaptureOwnership();

  CudaGraphSet cuda_graph_set_;
  CudaGraphAnnotation_t cuda_graph_annotation_id_ = kCudaGraphAnnotationDefault;

  cudaStream_t stream_ = nullptr;  // Does not own the stream
};

using CUDAGraph = CUDAGraphManager;

}  // namespace onnxruntime
