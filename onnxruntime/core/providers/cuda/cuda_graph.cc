#include "core/providers/cuda/cuda_graph.h"

#include "core/providers/cuda/cuda_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace onnxruntime {

CUDAGraph::CUDAGraph(cudaStream_t stream) : capture_stream_(stream) {
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11000)
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 11.0");
#endif
}

void CUDAGraph::CaptureBegin() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORT_ENFORCE(!has_graph_exec_,
              "This cuda graph has already captured a graph. "
              "Create a new instance to capture a new graph.");

  cudaDeviceSynchronize();
  CUDA_CALL_THROW(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 11.0");
#endif
}

void CUDAGraph::CaptureEnd() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CUDA_CALL_THROW(cudaStreamEndCapture(capture_stream_, &graph_));
  if (graph_ == NULL) {
    ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
  }

  has_graph_ = true;
  CUDA_CALL_THROW(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  has_graph_exec_ = true;
  CUDA_CALL_THROW(cudaGraphDestroy(graph_));
  has_graph_ = false;
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 11.0");
#endif
}

void CUDAGraph::Replay() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CUDA_CALL_THROW(cudaGraphLaunch(graph_exec_, capture_stream_));
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 11.0");
#endif
}

void CUDAGraph::Reset() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (has_graph_) {
    CUDA_CALL_THROW(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    CUDA_CALL_THROW(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
  is_capturing_ = false;
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 11.0");
#endif
}

bool CUDAGraph::IsCapturing() const {
  return is_capturing_;
}

void CUDAGraph::TurnOnCapture() {
  is_capturing_ = true;
}

void CUDAGraph::TurnOffCapture() {
  is_capturing_ = false;
}

CUDAGraph::~CUDAGraph() {
  Reset();
}

} // namespace onnxruntime
