// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/optimizer/cuda_runtime_fusion.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <functional>

#include "core/framework/compute_capability.h"
#include "core/optimizer/pattern_rewriter.h"

namespace onnxruntime::cuda_runtime_fusion::detail {

Status ProbeTarget(Graph& graph, const rewrite::Replacement& recipe,
                   const rewrite::Replacement::TensorBindings& bindings,
                   const NodeAttributes& attributes, size_t output_count,
                   const IExecutionProvider::IKernelLookup& lookup, bool& available);

using OptimizationFunction =
    std::function<Status(Graph&, const ComputeCapability&, ComputeCapability&, const GraphOptimizerRegistry&)>;

// Construct the offered region and callback holder without interpreting or rewriting the candidate.
std::unique_ptr<ComputeCapability> MakeCapability(gsl::span<const NodeIndex> source_ids,
                                                  OptimizationFunction optimization_func);

}  // namespace onnxruntime::cuda_runtime_fusion::detail

#endif
