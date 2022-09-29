// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/xnnpack/xnnpack_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
class GraphViewer;
namespace xnnpack {

class Concat final : public XnnpackKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;
  static bool IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph);

 private:
  OpComputeType op_type_ = OpComputeType::op_compute_type_invalid;
  InlinedVector<uint32_t> external_tensors_;
  XnnpackSubgraph subgraph_;
  XnnpackRuntime runtime_;
  XnnpackWorkspace workspace_;
};

class Split final : public XnnpackKernel, public SplitBase {
 public:
  Split(const OpKernelInfo& info);
  static bool IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph);

  Status Compute(OpKernelContext* context) const override;

 public:
  OpComputeType op_type_ = OpComputeType::op_compute_type_invalid;
  InlinedVector<uint32_t> external_tensors_;
  XnnpackSubgraph subgraph_;
  XnnpackRuntime runtime_;
  XnnpackWorkspace workspace_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
