// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/inlined_containers.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Accepted values of the kOrtSessionOptionsGqaValueLayout session option.
constexpr const char* kGqaValueLayoutBNSH = "BNSH";
constexpr const char* kGqaValueLayoutBNHS = "BNHS";

/**
The application-visible boundaries whose Value cache GqaValueLayoutTransformer converted to BNHS.

Recorded during the transform so that, after graph partitioning, the caller can tell which of the
inserted Transpose nodes an execution provider left behind. Graph input and output names are stable
across partitioning, which is what makes them usable as the anchor.
*/
struct GqaValueLayoutBoundaries {
  InlinedVector<std::string> past_value_inputs;      // graph inputs now declaring BNHS
  InlinedVector<std::string> present_value_outputs;  // graph outputs now declaring BNHS

  bool Empty() const { return past_value_inputs.empty() && present_value_outputs.empty(); }
};

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
  // converted_boundaries, when provided, collects the graph inputs and outputs this run converted,
  // for ReportUnfusedGqaValueLayoutTransposes() to check after partitioning.
  explicit GqaValueLayoutTransformer(GqaValueLayoutBoundaries* converted_boundaries = nullptr) noexcept
      : GraphTransformer("GqaValueLayoutTransformer"), converted_boundaries_(converted_boundaries) {
  }

  // Note: ShouldOnlyApplyOnce() is deliberately not overridden. Re-running must be safe anyway,
  // because a model saved with session.optimized_model_filepath already carries the transform and
  // may be reloaded into a new session with the option still set. The operand classification in
  // ApplyImpl is what provides that guarantee, and leaving this at the default keeps it under test.

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  GqaValueLayoutBoundaries* const converted_boundaries_;
};

/**
Finds the application boundaries of a graph that already carry the BNHS conversion.

For a graph this transformer did not process in this session -- an ORT format model converted offline,
for instance -- so that the same post-partition diagnostic can run over it. Call before partitioning,
while the GroupQueryAttention nodes are still there to anchor on.
*/
GqaValueLayoutBoundaries FindConvertedGqaValueLayoutBoundaries(const Graph& graph);

/**
Reports the converted boundaries whose Value-layout Transpose survived graph partitioning, i.e. that
will execute at runtime rather than having been fused away. Logs a warning naming each one and
returns their names. Call after partitioning, and only when the BNHS layout was requested.

Anchored on the boundaries rather than on the GroupQueryAttention nodes on purpose. A compiling EP
may claim the GQA node and replace it with a fused node while leaving the flanking Transposes in the
graph; both full-cache copies still execute, but there is no GQA node left to search from.

Without this, an EP that silently declines to fuse turns into a large per-step cost with nothing in
the logs to explain it.
*/
InlinedVector<std::string> ReportUnfusedGqaValueLayoutTransposes(const Graph& graph,
                                                                 const GqaValueLayoutBoundaries& boundaries,
                                                                 const logging::Logger& logger);

}  // namespace onnxruntime
