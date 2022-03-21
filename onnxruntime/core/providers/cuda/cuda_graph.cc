// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_graph.h"

#include "core/providers/cuda/cuda_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace onnxruntime {

CUDAGraph::CUDAGraph(cudaStream_t stream) : stream_(stream) {
#if (defined(CUDA_VERSION) && CUDA_VERSION < 10000)
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
#endif
}

void CUDAGraph::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CUDAGraph::CaptureBegin() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  ORT_ENFORCE(!has_graph_exec_,
              "This cuda graph has already captured a graph. "
              "Create a new instance to capture a new graph.");

  CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
  // For now cuda graph can only work with a single thread. In the future, we
  // will support multiple threads. For multiple threads with multiple graphs
  // and streams, `cudaStreamCaptureModeGlobal` needs to be changed to
  // `cudaStreamCaptureModeThreadLocal`
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
#endif
}

void CUDAGraph::CaptureEnd() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph_));
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
  // Although this function is not thread safe, the lock is not needed here because
  // CUDA EP maintains a separate cuda graph per thread
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  LOGS_DEFAULT(INFO) << "Replaying CUDA graph on stream " << stream_;
  CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph_exec_, stream_));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
#else
  ORT_THROW("CUDA graphs can only be used in Onnxruntime built with CUDA >= 10.0");
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

CUDAGraph::~CUDAGraph() {
  Reset();
}

} // namespace onnxruntime
