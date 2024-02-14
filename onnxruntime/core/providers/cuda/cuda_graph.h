// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CaptureId_t = unsigned long long;

struct CUDAGraph {
  CUDAGraph(){};
  CUDAGraph(cudaStream_t stream);
  ~CUDAGraph();

  void SetStream(cudaStream_t stream);
  void CaptureBegin(optional<int> cuda_graph_annotation_id);
  void CaptureEnd();
  Status Replay(optional<int> cuda_graph_annotation_id);

  void Reset();
  void ResetAdditional();

 private:
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;

  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  cudaGraph_t additional_graph_ = NULL;
  std::unordered_map<int, cudaGraphExec_t> graph_exec_map_;
  optional<int> cuda_graph_annotation_id_;
  bool has_additional_graph_ = false;

  cudaStream_t stream_ = nullptr;  // Does not own the stream
};

}  // namespace onnxruntime
