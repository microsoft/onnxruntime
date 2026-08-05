// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// This file is the ROCm equivalent of cuda_graph.h, providing the
// RocmGraphAnnotation_t type alias and ROCMGraph (= ROCMGraphManager) class.

#pragma once

#include <unordered_map>

#include "core/common/common.h"
#include <mutex>
#include "core/providers/rocm/rocm_pch.h"

namespace onnxruntime {

using RocmGraphAnnotation_t = int;
using RocmGraphSet_t = std::unordered_map<RocmGraphAnnotation_t, hipGraphExec_t>;

constexpr RocmGraphAnnotation_t kRocmGraphAnnotationSkip = -1;
constexpr RocmGraphAnnotation_t kRocmGraphAnnotationDefault = 0;

struct RocmGraphSet {
  RocmGraphSet() {}
  ~RocmGraphSet();

  void Clear();
  bool Contains(RocmGraphAnnotation_t rocm_graph_annotation_id) const;
  void Put(RocmGraphAnnotation_t rocm_graph_annotation_id, hipGraphExec_t graph_exec);
  hipGraphExec_t Get(RocmGraphAnnotation_t rocm_graph_annotation_id) const;

 private:
  RocmGraphSet_t rocm_graphs_;
};

struct ROCMGraphManager {
  ROCMGraphManager() {}
  explicit ROCMGraphManager(hipStream_t stream);
  ~ROCMGraphManager();

  void SetStream(hipStream_t stream);
  void CaptureBegin(RocmGraphAnnotation_t rocm_graph_annotation_id);
  void CaptureEnd(RocmGraphAnnotation_t rocm_graph_annotation_id);
  Status Replay(RocmGraphAnnotation_t rocm_graph_annotation_id, bool sync_status_flag = true);

  void Reset();

  bool IsGraphCaptureAllowedOnRun(RocmGraphAnnotation_t rocm_graph_annotation_id) const;
  bool IsGraphCaptured(RocmGraphAnnotation_t rocm_graph_annotation_id) const;

 private:
  RocmGraphSet rocm_graph_set_;
  RocmGraphAnnotation_t rocm_graph_annotation_id_ = kRocmGraphAnnotationDefault;

  hipStream_t stream_ = nullptr;  // Does not own the stream
};

using ROCMGraph = ROCMGraphManager;

}  // namespace onnxruntime
