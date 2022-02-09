#include "core/providers/cuda/cuda_graph.h"

#include "core/providers/cuda/cuda_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace onnxruntime {

CUDAGraph::CUDAGraph(cudaStream_t stream) : capture_stream_(stream) {
#if (defined(CUDA_VERSION) && CUDA_VERSION < 10000)
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 11.0");
#endif
}

void CUDAGraph::CaptureBegin() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  ORT_ENFORCE(!has_graph_exec_,
              "This cuda graph has already captured a graph. "
              "Create a new instance to capture a new graph.");

  CUDA_CALL_THROW(cudaDeviceSynchronize());
  CUDA_CALL_THROW(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
#endif
}

void CUDAGraph::CaptureEnd() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
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
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
#endif
}

Status CUDAGraph::Replay() {
  std::lock_guard<OrtMutex> lock(lock_);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph_exec_, capture_stream_));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(capture_stream_));

  // TODO: check if the cuda graph issue mentioned in pytorch can be resolved by
  // doing stream synchronize here instead of cudaDeviceSynchronize. The solution
  // in pytorch can be found in below:
  //
  // int version;
  // CUDA_RETURN_IF_ERROR(cudaDriverGetVersion(&version));
  // if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    // CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
  // }
#else
  CUDA_RETURN_IF_ERROR("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
#endif
  return Status::OK();
}

void CUDAGraph::Reset() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  if (has_graph_) {
    CUDA_CALL_THROW(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    CUDA_CALL_THROW(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
#endif
}

void CUDAGraph::SetStream(cudaStream_t stream) {
  capture_stream_ = stream;
}

CUDAGraph::~CUDAGraph() {
  Reset();
}

} // namespace onnxruntime
