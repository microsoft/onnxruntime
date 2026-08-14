// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_graph.h"

#include "core/providers/cuda/cuda_common.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace onnxruntime {

namespace {

// ORT-initiated captures are tracked per stream, process wide, rather than per manager. A manager
// belongs to one (execution provider, thread) pair while a stream may be shared by several sessions
// and driven from several threads, so a per-manager flag would let a manager that did not start the
// capture see "stream is capturing" with its own flag clear, conclude the capture belongs to the
// caller, silently skip execution, and return stale outputs.
std::mutex& OrtCaptureOwnerMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<cudaStream_t, const CUDAGraphManager*>& OrtCaptureOwners() {
  static std::unordered_map<cudaStream_t, const CUDAGraphManager*> owners;
  return owners;
}

}  // namespace

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

void CudaGraphSet::Clear() {
  for (auto& it : cuda_graphs_) {
    // Reached from ~CudaGraphSet(), so this must not throw: a CUDA error here (for example when
    // teardown races a stream that is still capturing) would otherwise terminate the process.
    cudaError_t status = cudaGraphExecDestroy(it.second);
    if (status != cudaSuccess) {
      LOGS_DEFAULT(ERROR) << "Failed to destroy a captured CUDA graph: " << cudaGetErrorString(status);
      cudaGetLastError();  // clear the sticky error so it is not misattributed later
    }
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

bool CUDAGraphManager::IsStreamCapturing(cudaStream_t stream) {
  // The legacy default stream is never capturable, and capture state is only meaningful for a
  // real stream handle.
  if (stream == nullptr) {
    return false;
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  // Query, never throw: this runs on hot paths that must stay usable even if the query fails,
  // in which case the pre-existing (non-capturing) behavior is the safe answer.
  if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess) {
    // Clear the sticky error so the failed query does not surface later as an unrelated failure.
    cudaGetLastError();
    return false;
  }

  return capture_status != cudaStreamCaptureStatusNone;
}

bool CUDAGraphManager::IsExternalCaptureActive() const {
  if (!IsStreamCapturing(stream_)) {
    return false;
  }

  // A capture that any ORT manager started on this stream is ORT's own, no matter which manager or
  // thread observes it.
  std::lock_guard<std::mutex> lock(OrtCaptureOwnerMutex());
  return OrtCaptureOwners().find(stream_) == OrtCaptureOwners().end();
}

void CUDAGraphManager::CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  ORT_ENFORCE(IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id));

  ORT_ENFORCE(!cuda_graph_set_.Contains(cuda_graph_annotation_id),
              "Trying to capture a graph with annotation id ", cuda_graph_annotation_id,
              " that already used. Please use a different annotation id.");

  // A capture cannot be nested inside another capture on the same stream. Report the situation
  // instead of letting cudaStreamSynchronize/cudaStreamBeginCapture fail with a bare CUDA error.
  ORT_ENFORCE(!IsStreamCapturing(stream_),
              "Cannot start an ONNX Runtime CUDA graph capture on stream ", stream_,
              " because a device graph capture is already in progress on it. When the caller "
              "captures its own graph around Run(), ORT records the run into that graph instead "
              "of capturing one itself; this run reached ORT-managed capture unexpectedly.");

  CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
  // For now cuda graph can only work with a single thread. In the future, we
  // will support multiple threads. For multiple threads with multiple graphs
  // and streams, `cudaStreamCaptureModeGlobal` needs to be changed to
  // `cudaStreamCaptureModeThreadLocal`
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  {
    // The stream was verified not to be capturing just above, so any record still present is stale
    // (a previous run that failed between begin and end) and is replaced here.
    std::lock_guard<std::mutex> lock(OrtCaptureOwnerMutex());
    OrtCaptureOwners()[stream_] = this;
  }
}

void CUDAGraphManager::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id) {
  cudaGraph_t graph = NULL;
  // Release ownership first so a failure below cannot leave the stream permanently marked as
  // ORT-captured, which would make a later caller-initiated capture look like ORT's own.
  ReleaseOrtCaptureOwnership();
  CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph));
  if (graph == NULL) {
    ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
  }

  cudaGraphExec_t graph_exec = NULL;
  CUDA_CALL_THROW(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  CUDA_CALL_THROW(cudaGraphDestroy(graph));

  // Currently all the captured graphs will be tied to the session's lifecycle
  // TODO(wy): Addd an interface to free captured graphs
  cuda_graph_set_.Put(cuda_graph_annotation_id, graph_exec);
}

Status CUDAGraphManager::Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync_status_flag) {
  // Although this function is not thread safe, the lock is not needed here because
  // CUDA EP maintains a separate cuda graph per thread
  LOGS_DEFAULT(INFO) << "Replaying CUDA graph on stream " << stream_ << " with cuda_graph_annotation_id "
                     << cuda_graph_annotation_id;

  // cudaGraphLaunch is rejected on a capturing stream, so a replay requested while the caller is
  // capturing means the caller's graph would silently miss this work. Fail with an explanation
  // rather than surfacing CUDA error 900 from deep inside the launch.
  if (IsStreamCapturing(stream_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Cannot replay an ONNX Runtime CUDA graph on stream ", stream_,
                           " while a device graph capture is in progress on it. CUDA does not allow "
                           "launching a graph into a capturing stream. Runs issued inside a "
                           "caller-initiated capture are recorded rather than replayed.");
  }

  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(cuda_graph_annotation_id);
  CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph_exec, stream_));

  if (sync_status_flag) {
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
  }
  return Status::OK();
}

bool CUDAGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_annotation_id != kCudaGraphAnnotationSkip;
}

bool CUDAGraphManager::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const {
  return cuda_graph_set_.Contains(cuda_graph_annotation_id);
}

void CUDAGraphManager::Reset() {
  cuda_graph_set_.Clear();
}

void CUDAGraphManager::ReleaseOrtCaptureOwnership() {
  std::lock_guard<std::mutex> lock(OrtCaptureOwnerMutex());
  auto it = OrtCaptureOwners().find(stream_);
  if (it != OrtCaptureOwners().end() && it->second == this) {
    OrtCaptureOwners().erase(it);
  }
}

CUDAGraphManager::~CUDAGraphManager() {
  // A manager destroyed mid-capture (for example while unwinding from a failed run) must not leave
  // the stream marked as ORT-captured.
  ReleaseOrtCaptureOwnership();
  Reset();
}

}  // namespace onnxruntime
