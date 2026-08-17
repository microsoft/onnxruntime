// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Accepted values of the kOrtSessionOptionsGqaValueLayout session option.
constexpr const char* kGqaValueLayoutBNSH = "BNSH";
constexpr const char* kGqaValueLayoutBNHS = "BNHS";

/**
@class GqaValueLayoutTransformer

Adapts com.microsoft.GroupQueryAttention nodes to a BNHS Value KV-cache at the graph boundary.

The GQA operator schema requires the Value cache in BNSH layout
(batch_size, num_heads, sequence_length, head_size). Some execution providers execute the operator
more efficiently when the application holds that cache as BNHS
(batch_size, num_heads, head_size, sequence_length) instead.

When the application selects BNHS via the kOrtSessionOptionsGqaValueLayout session option, this
transformer keeps the GQA node itself in BNSH and moves the conversion into the graph:

    past_value (BNHS, graph input) -> Transpose[0,1,3,2] -> GQA -> Transpose[0,1,3,2] -> present_value (BNHS, graph output)

The declared shapes of the past_value graph input and present_value graph output are updated to
BNHS so that session input/output validation accepts the application's buffers.

An EP that prefers BNHS is expected to fuse the Transpose -> GQA -> Transpose sequence into a single
operation, making the transposes free. An EP that does not fuse them executes them, which is correct
but costs a full copy of the Value cache in each direction per step.

Only the main graph is processed; the Key cache is not affected.
*/
class GqaValueLayoutTransformer : public GraphTransformer {
 public:
  GqaValueLayoutTransformer() noexcept
      : GraphTransformer("GqaValueLayoutTransformer") {
  }

  // Note: ShouldOnlyApplyOnce() is deliberately not overridden. Re-running must be safe anyway,
  // because a model saved with session.optimized_model_filepath already carries the transform and
  // may be reloaded into a new session with the option still set. AlreadyTransformed() is what
  // provides that guarantee, and leaving this at the default keeps it under test.

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

/**
Logs a warning for every Value-layout Transpose that survived graph partitioning, i.e. that no
execution provider fused into its GroupQueryAttention node and that will therefore execute at
runtime. Call after partitioning, and only when the BNHS layout was requested.

Without this, an EP that silently declines to fuse turns into a large per-step cost with nothing in
the logs to explain it.
*/
void LogUnfusedGqaValueLayoutTransposes(const Graph& graph, const logging::Logger& logger);

}  // namespace onnxruntime
