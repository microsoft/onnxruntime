#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CaptureId_t = unsigned long long;

struct CUDAGraph {
  CUDAGraph(cudaStream_t stream);
  ~CUDAGraph();

  void CaptureBegin();
  void CaptureEnd();
  void Replay();
  void Reset();
  bool IsCapturing() const;
  void TurnOnCapture();
  void TurnOffCapture();

  protected:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;
#endif

  bool has_graph_ = false;
  bool has_graph_exec_ = false;
  bool is_capturing_ = false;

  CaptureId_t id_;
  cudaStream_t capture_stream_ = nullptr;
  };
 
} // namespace onnxruntime
