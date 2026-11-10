// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/optimizer/pattern_rewriter.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
namespace onnxruntime {
class GraphViewer;
}

namespace onnxruntime::contrib::cuda::fusion::internal {

// Necessary topology filter only; the existing matcher remains authoritative.
InlinedVector<NodeIndex> CollectMoERoots(const GraphViewer& viewer);

// Fixed-(E,k) structural regression helper. The production declaration shares its
// topology and semantic checks, but captures route_k and selects E by alternative.
class MoEStructuralCandidate final {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MoEStructuralCandidate);
  MoEStructuralCandidate(int experts, int k);
  const rewrite::Rule& Rule() const { return rule_; }

 private:
  const rewrite::Rule rule_;
};

}  // namespace onnxruntime::contrib::cuda::fusion::internal
#endif
