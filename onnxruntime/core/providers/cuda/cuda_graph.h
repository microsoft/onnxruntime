// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CaptureId_t = unsigned long long;

struct CUDAGraph {
  CUDAGraph(){};
  CUDAGraph(cudaStream_t stream);
  ~CUDAGraph();

  void SetStream(cudaStream_t stream);
  void CaptureBegin();
  void CaptureEnd();
  Status Replay();
  void Reset();

 private:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  cudaStream_t stream_ = nullptr;  // Does not own the stream
};

}  // namespace onnxruntime
