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
 * data-layout preference for layout-sensitive EPs (e.g. NHWC), ahead-of-time/
 * just-in-time compilation of fused subgraphs, and TunableOp tuning. Bundling
 * them on the base couples every EP and every caller to the union of all
 * capabilities.
 *
 * Each mix-in below groups one such cluster behind a narrow interface. An EP
 * that supports a capability implements the corresponding mix-in and returns it
 * from the matching IExecutionProvider::Get*Capability() query hook; callers
 * depend only on the mix-in they actually use.
 *
 * The goal is explicit optional-capability discovery and narrower caller
 * dependencies, not reduced memory use or improved runtime performance. Opting
 * into a mix-in adds polymorphic subobjects/vtables and pointer adjustment.
 * Returning this through the declared capability type uses the compiler's
 * derived-to-base conversion; callers must not assume identical EP and
 * capability addresses or reinterpret_cast between them.
 *
 * Legacy method signatures and source behavior are retained, and discovery
 * hooks default to nullptr until an EP opts in. This is not a binary ABI
 * compatibility guarantee: internal interface/layout changes require matching
 * EP and runtime rebuilds. Production migration and any removal of legacy
 * virtuals require a separate owner-approved compatibility decision.
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
   *             The default (true) must stay equal to IExecutionProvider::ReplayGraph's
   *             default: a default argument binds by the static type of the call
   *             expression, so the two declarations must agree to behave identically
   *             regardless of which one a caller sees.
   */
  virtual common::Status ReplayGraph(int graph_annotation_id, bool sync = true) = 0;

  /**
   * Release a previously captured graph and its associated resources.
   *
   * Thread safety: for EPs where ConcurrentRunSupported() returns true, this may be
   * called concurrently with Run(), and the EP is responsible for its own
   * synchronization; for non-concurrent EPs the session serializes the calls.
   */
  virtual common::Status ReleaseCapturedGraph(int graph_annotation_id) = 0;

  /** Get the node assignment validation policy to apply when graph capture is enabled. */
  virtual OrtGraphCaptureNodeAssignmentPolicy GetGraphCaptureNodeAssignmentPolicy() const = 0;
};

/**
 * TunableOp tuning capability.
 *
 * Mirrors the legacy IExecutionProvider::GetTuningContext() signature, including
 * its mutable ITuningContext* return from a const accessor. Constness is shallow:
 * a const capability or EP does not make the returned context or tuning state
 * immutable. This retains the existing contract rather than introducing deep
 * constness or new thread-safety guarantees. Context mutation still requires
 * the synchronization prescribed by the context/provider.
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
   *
   * Do NOT cache the GraphViewer in FusedNodeAndGraph.filtered_graph in any of the NodeComputeInfo
   * functions, as it is only valid for the duration of the call to Compile.
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
