// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

using CaptureId_t = unsigned long long;
using GraphAnnotation_t = int;
using GraphAnnotationOptional_t = optional<GraphAnnotation_t>;

struct CUDAGraph {
  CUDAGraph(){};
  CUDAGraph(cudaStream_t stream);
  ~CUDAGraph();

  void SetStream(cudaStream_t stream);
  // bugbug: handle -1 here
  void CaptureBegin(GraphAnnotationOptional_t cuda_graph_annotation_id);
  void CaptureEnd();
  Status Replay(GraphAnnotationOptional_t cuda_graph_annotation_id);

  void Reset();
  void ResetAdditional();

  bool IsAdditionalGraphCaptured() const;

 private:
  cudaGraph_t graph_ = NULL;
  cudaGraphExec_t graph_exec_ = NULL;

  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  cudaGraph_t additional_graph_ = NULL;
  std::unordered_map<GraphAnnotation_t, cudaGraphExec_t> graph_exec_map_;
  GraphAnnotationOptional_t cuda_graph_annotation_id_;
  bool has_additional_graph_ = false;

  cudaStream_t stream_ = nullptr;  // Does not own the stream
};

}  // namespace onnxruntime
