// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CudaGraphAnnotation_t = int;
using CudaGraphSet_t = std::unordered_map<CudaGraphAnnotation_t, cudaGraphExec_t>;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

struct CudaGraphSet {
  CudaGraphSet(){};
  ~CudaGraphSet();

  void Clear();
  bool Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
  void Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec);
  cudaGraphExec_t Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

 private:
  CudaGraphSet_t cuda_graphs_;
};

struct CUDAGraphManager {
  CUDAGraphManager(){};
  CUDAGraphManager(cudaStream_t stream);
  ~CUDAGraphManager();

  void SetStream(cudaStream_t stream);
  void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
  void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
  Status Replay(CudaGraphAnnotation_t cuda_graph_annotation_id);

  void Reset();

  bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
  bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

 private:
  CudaGraphSet cuda_graph_set_;
  CudaGraphAnnotation_t cuda_graph_annotation_id_ = kCudaGraphAnnotationDefault;

  cudaStream_t stream_ = nullptr;  // Does not own the stream
};

using CUDAGraph = CUDAGraphManager;

}  // namespace onnxruntime
