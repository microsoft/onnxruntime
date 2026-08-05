// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_graph.h"

#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {

RocmGraphSet::~RocmGraphSet() {
  Clear();
}

void RocmGraphSet::Clear() {
  for (auto& it : rocm_graphs_) {
    HIP_CALL_THROW(hipGraphExecDestroy(it.second));
  }
  rocm_graphs_.clear();
}

bool RocmGraphSet::Contains(RocmGraphAnnotation_t rocm_graph_annotation_id) const {
  return rocm_graphs_.find(rocm_graph_annotation_id) != rocm_graphs_.end();
}

void RocmGraphSet::Put(RocmGraphAnnotation_t rocm_graph_annotation_id, hipGraphExec_t graph_exec) {
  ORT_ENFORCE(!Contains(rocm_graph_annotation_id));
  rocm_graphs_.emplace(rocm_graph_annotation_id, graph_exec);
}

hipGraphExec_t RocmGraphSet::Get(RocmGraphAnnotation_t rocm_graph_annotation_id) const {
  ORT_ENFORCE(Contains(rocm_graph_annotation_id));
  return rocm_graphs_.at(rocm_graph_annotation_id);
}

ROCMGraphManager::ROCMGraphManager(hipStream_t stream) : stream_(stream) {
}

void ROCMGraphManager::SetStream(hipStream_t stream) {
  stream_ = stream;
}

void ROCMGraphManager::CaptureBegin(RocmGraphAnnotation_t rocm_graph_annotation_id) {
  ORT_ENFORCE(IsGraphCaptureAllowedOnRun(rocm_graph_annotation_id));

  ORT_ENFORCE(!rocm_graph_set_.Contains(rocm_graph_annotation_id),
              "Trying to capture a graph with annotation id ", rocm_graph_annotation_id,
              " that already used. Please use a different annotation id.");

  HIP_CALL_THROW(hipStreamSynchronize(stream_));
  HIP_CALL_THROW(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
}

void ROCMGraphManager::CaptureEnd(RocmGraphAnnotation_t rocm_graph_annotation_id) {
  hipGraph_t graph = NULL;
  HIP_CALL_THROW(hipStreamEndCapture(stream_, &graph));
  if (graph == NULL) {
    ORT_THROW("ROCMGraph::CaptureEnd: graph_ is NULL");
  }

  hipGraphExec_t graph_exec = NULL;
  HIP_CALL_THROW(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  HIP_CALL_THROW(hipGraphDestroy(graph));

  rocm_graph_set_.Put(rocm_graph_annotation_id, graph_exec);
}

Status ROCMGraphManager::Replay(RocmGraphAnnotation_t rocm_graph_annotation_id, bool sync_status_flag) {
  LOGS_DEFAULT(INFO) << "Replaying ROCm graph on stream " << stream_ << " with rocm_graph_annotation_id "
                     << rocm_graph_annotation_id;

  hipGraphExec_t graph_exec = rocm_graph_set_.Get(rocm_graph_annotation_id);
  HIP_RETURN_IF_ERROR(hipGraphLaunch(graph_exec, stream_));

  if (sync_status_flag) {
    HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream_));
  }
  return Status::OK();
}

bool ROCMGraphManager::IsGraphCaptureAllowedOnRun(RocmGraphAnnotation_t rocm_graph_annotation_id) const {
  return rocm_graph_annotation_id != kRocmGraphAnnotationSkip;
}

bool ROCMGraphManager::IsGraphCaptured(RocmGraphAnnotation_t rocm_graph_annotation_id) const {
  return rocm_graph_set_.Contains(rocm_graph_annotation_id);
}

void ROCMGraphManager::Reset() {
  rocm_graph_set_.Clear();
}

ROCMGraphManager::~ROCMGraphManager() {
  Reset();
}

}  // namespace onnxruntime
