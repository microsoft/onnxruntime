// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/status.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

class GraphViewer;

/**
 * Segregated optional-capability interfaces for IExecutionProvider.
 *
 * Motivation (Interface Segregation Principle):
 * IExecutionProvider historically carries a large number of defaulted virtual
 * methods, many of which represent optional capabilities relevant to only a
 * subset of execution providers -- graph capture/replay (e.g. CUDA graphs),
 * ahead-of-time/just-in-time compilation of fused subgraphs, and TunableOp
 * tuning. Bundling them on the base couples every EP and every caller to the
 * union of all capabilities.
 *
 * Each mix-in below groups one such cluster behind a narrow interface. An EP
 * that supports a capability implements the corresponding mix-in and returns it
 * from the matching IExecutionProvider::Get*Capability() query hook; callers
 * depend only on the mix-in they actually use.
 *
 * This is additive and non-breaking: the legacy per-capability virtuals on
 * IExecutionProvider remain in place, so existing EPs and callers are
 * unaffected. EPs and callers can migrate to the segregated interfaces
 * incrementally, and a cluster's legacy virtuals can be removed from the base
 * once all of its implementers and callers have moved over.
 */

/**
 * Graph capture / replay capability (e.g. CUDA graphs).
 *
 * Method signatures mirror the corresponding legacy IExecutionProvider virtuals
 * so that a migrating EP can satisfy both with a single set of definitions.
 */
class IGraphCaptureCapability {
 public:
  virtual ~IGraphCaptureCapability() = default;

  /** Indicate whether graph capture/replay is enabled for the provider. */
  virtual bool IsGraphCaptureEnabled() const = 0;

  /** Indicate whether the graph for the given annotation id has been captured and instantiated. */
  virtual bool IsGraphCaptured(int graph_annotation_id) const = 0;

  /**
   * Run the instantiated graph.
   * @param sync If true, synchronize the device/stream after replay before returning.
   */
  virtual common::Status ReplayGraph(int graph_annotation_id, bool sync = true) = 0;

  /** Release a previously captured graph and its associated resources. */
  virtual common::Status ReleaseCapturedGraph(int graph_annotation_id) = 0;

  /** Get the node assignment validation policy to apply when graph capture is enabled. */
  virtual OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const = 0;
};

/**
 * TunableOp tuning capability.
 */
class ITuningCapability {
 public:
  virtual ~ITuningCapability() = default;

  /** Return the tuning context which holds all TunableOp state. */
  virtual ITuningContext* GetTuningContext() const = 0;
};

/**
 * Data-layout preference capability.
 *
 * Only a subset of EPs prefer a non-default (non-NCHW) data layout. Such an EP
 * advertises its preferred layout and decides, per op, whether ORT should
 * convert an associated node's data layout during layout transformation. The
 * two methods are coupled: ShouldConvertDataLayoutForOp is driven by the
 * preferred layout reported by GetPreferredLayout.
 */
class IDataLayoutCapability {
 public:
  virtual ~IDataLayoutCapability() = default;

  /** Return the data layout preferred by this EP. */
  virtual DataLayout GetPreferredLayout() const = 0;

  /**
   * Decide whether an op (with the given `domain` and `op_type`) should have its
   * data layout converted to `target_data_layout`. Return std::nullopt to leave
   * the decision to ORT.
   */
  virtual std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view domain,
                                                           std::string_view op_type,
                                                           DataLayout target_data_layout) const = 0;
};

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
/**
 * Subgraph-compilation capability.
 *
 * Mirrors the legacy compilation virtuals, which are likewise only available
 * outside a (non-extended) minimal build.
 */
class ICompileCapability {
 public:
  virtual ~ICompileCapability() = default;

  /**
   * Given a collection of fused Nodes and the respective GraphViewer instance for the nodes that were
   * fused, return create_state/compute/release_state func for each node.
   */
  virtual common::Status Compile(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
                                 std::vector<NodeComputeInfo>& node_compute_funcs) = 0;

  /** Get the compatibility info for a compiled model. */
  virtual std::string GetCompiledModelCompatibilityInfo(const GraphViewer& graph_viewer) const = 0;

  /** Validate the compatibility of a compiled model with this execution provider. */
  virtual common::Status ValidateCompiledModelCompatibilityInfo(
      const std::string& compatibility_info,
      OrtCompiledModelCompatibility& model_compatibility) const = 0;
};
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

}  // namespace onnxruntime
