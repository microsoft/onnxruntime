#pragma once

#include <cstddef>
#include <memory>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {

class Graph;
class Model;

#if !defined(ORT_MINIMAL_BUILD)

/**
 * Resource limits for one FunctionExtractor call.
 *
 * A zero limit permits no corresponding work and returns an error before the
 * current replacement batch mutates the graph.
 */
struct FunctionExtractorOptions {
  /** Maximum total FunctionProto body nodes, including Constant nodes. */
  size_t max_pattern_nodes{1024};
  /** Maximum nodes in the single target Graph scope being searched. */
  size_t max_target_nodes{1'000'000};
  /** Maximum candidate-root entries and output-root tuples per discovery pass. */
  size_t max_output_root_tuples{100'000};
  /** Maximum pattern-slot normalization and aggregate matcher work per pass. */
  size_t max_worklist_bindings{1'000'000};
  /** Maximum aggregate literal bytes normalized or compared per pass. */
  size_t max_literal_bytes{64U * 1024U * 1024U};
};

/**
 * Result of extraction.
 *
 * replacements_applied includes calls added before a later post-mutation
 * failure. Such a failure may leave the graph modified and unresolved.
 */
struct FunctionExtractionResult {
  common::Status status{common::Status::OK()};
  size_t replacements_applied{0};
};

class FunctionExtractor final {
 public:
  /**
   * Copies and validates function_proto. The caller may destroy the source
   * proto after construction.
   *
   * V1 supports connected, acyclic, pure tensor DAGs with fixed attributes,
   * positional optional/variadic slots, and dense Constant literals. It does
   * not support function attribute references, required call attributes,
   * control-flow bodies, or provider-assigned matches.
   */
  explicit FunctionExtractor(
      const ONNX_NAMESPACE::FunctionProto& function_proto,
      FunctionExtractorOptions options = {});
  ~FunctionExtractor();

  /**
   * Extracts to fixpoint from model.MainGraph(). The identical function
   * definition must already be registered in the model.
   */
  FunctionExtractionResult Extract(Model& model);

  /**
   * Extracts to fixpoint from one graph scope.
   *
   * The graph must already be resolved, with a schema on every node, and its
   * owning model must pre-register an identical function definition. Success
   * guarantees a resolved graph at fixpoint. Validation/discovery failures do
   * not mutate the current batch; failures after mutation are non-atomic and
   * are reported with the exact replacements_applied count.
   */
  FunctionExtractionResult Extract(Graph& graph);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FunctionExtractor);
};

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
