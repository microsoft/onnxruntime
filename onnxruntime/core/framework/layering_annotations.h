// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/common/logging/logging.h"
#include "gsl/gsl"
#include <string>
#include <vector>
#include <optional>
#include <memory>

struct OrtEpDevice;

namespace onnxruntime {
class ExecutionProviders;
class Graph;

/// <summary>
/// Annotation extracted from kOrtSessionOptionsLayerAssignmentSettings session configuration option.
/// </summary>
struct LayerAnnotation {
  std::string device;
  std::string annotation;
  bool prefix_match;
};

/// <summary>
/// This struct is a container for layering rules extracted from the kOrtSessionOptionsLayerAssignmentSettings
/// session configuration option.
/// </summary>
struct LayeringRules {
  std::vector<LayerAnnotation> rules;
  /// <summary>
  /// Parses the layering rules from the given configuration string.
  /// The configuration string is in the following format.:
  /// 'cpu(L1,L2); gpu(L3,=L4)' where cpu or gpu denote the target EP.
  /// L1, L2, L3 are annotations that can be matched to node annotations in the graph. The '=' prefix denotes
  /// exact match. The position of the annotation (L1, L2, L3) in the list denotes its priority in matching (left to right).
  /// However, the prefix annotations will always have higher priority than the exact match annotations regardless
  /// of their position in the list. In the above example, L1 has the highest priority, followed by L2,
  /// then L3 and finally L4. The rules are separated by ';' and there can be multiple rules for different EPs.
  /// </summary>
  /// <param name="config_value">The configuration string to parse.</param>
  /// <param name="rules">Output parameter where the parsed rules will be stored.</param>
  /// <returns>Status indicating success or failure (e.g. due to format errors).</returns>
  static common::Status FromConfigString(const std::string& config_value, LayeringRules& rules);
};

/// <summary>
/// This class matches node annotations against layering rules.
/// </summary>
class LayeringRuleMatcher {
 public:
  explicit LayeringRuleMatcher(const LayeringRules& rules);

  /// <summary>
  /// The method returns the index of the best matching rule for the given annotation
  /// if it exists
  /// </summary>
  /// <param name="node_annotation">annotation retrieved from protobuf node metadata</param>
  /// <returns>index of the matching LayeringRule if it exists</returns>
  std::optional<size_t> Match(const std::string& node_annotation) const;

 private:
  struct TrieNode {
    InlinedHashMap<char, std::unique_ptr<TrieNode>> children;
    std::optional<size_t> rule_index;
  };

  TrieNode root_;
  InlinedHashMap<std::string, size_t> exact_match_rules_;

  void AddExactRule(const std::string& annotation, size_t index);

  void AddPrefixRule(const std::string& annotation, size_t index);

  void UpdateBestMatch(std::optional<size_t>& current_best, size_t candidate) const;
};

namespace EpLayeringMatcher {
/// <summary>
/// Matches a list of available OrtEpDevices against the device string specified in the LayerAnnotation.
/// Returns the EP Type string of the first device that matches the rule.
/// </summary>
/// <param name="ep_devices">The list of available EP devices.</param>
/// <param name="rule">The rule containing the device designator.</param>
/// <returns>Optional containing the matched EP type, nullopt otherwise.</returns>
std::optional<std::string> Match(gsl::span<const OrtEpDevice* const> ep_devices,
                                 const LayerAnnotation& rule);

/// <summary>
/// Matches a collection of ExecutionProviders against the device string specified in the LayerAnnotation.
/// Returns the EP Type string of the first provider that matches the rule.
/// </summary>
/// <param name="providers">The collection of available Execution Providers.</param>
/// <param name="rule">The rule containing the device designator.</param>
/// <returns>Optional containing the matched EP type, nullopt otherwise.</returns>
std::optional<std::string> Match(const ExecutionProviders& providers, const LayerAnnotation& rule);
}  // namespace EpLayeringMatcher

// This class contains indexing information about the entire graph
// per sub-graph info is stored in graph_index_
class LayeringIndex {
 public:
  // mapping of EP name/type to a set of LayeringRule indices mapped to that EP.
  using EpNameToLayeringIndices = InlinedHashMap<std::string, InlinedHashSet<size_t>>;
  // mapping of LayeringRule index to EP name/type, reverse of the above
  using LayeringIndexToEpName = InlinedHashMap<size_t, std::string>;

  /// <summary>
  /// Creates a fully initialized LayeringIndex.
  /// </summary>
  /// <param name="graph">The graph to traverse and index.</param>
  /// <param name="ep_map">Pre-populated mapping of EP names to their applicable rule indices.</param>
  /// <param name="rule_map">Pre-populated mapping of rule indices to EP names.</param>
  /// <param name="matcher">Matcher to resolve node annotations to rule indices.</param>
  static LayeringIndex Create(const Graph& graph,
                              EpNameToLayeringIndices ep_map,
                              LayeringIndexToEpName rule_map,
                              LayeringRules layering_rules);

  /// <summary>
  /// Factory method that creates a LayeringIndex by parsing configuration, matching rules against
  /// available devices/providers, and indexing the graph.
  /// </summary>
  /// <param name="graph">The graph to index.</param>
  /// <param name="config_string">The configuration string containing layering rules.</param>
  /// <param name="ep_devices">Available OrtEpDevices to match rules against.</param>
  /// <param name="ep_providers">Available ExecutionProviders to match rules against (fallback).</param>
  /// <param name="logger">Logger for reporting information/errors.</param>
  /// <param name="layering_index">Output parameter for the created LayeringIndex. Returns no index if
  ///              no valid layering rules discovered.</param>
  /// <returns>Status indicating success or failure.</returns>
  static Status Create(const Graph& graph,
                       const std::string& config_string,
                       gsl::span<const OrtEpDevice* const> ep_devices,
                       const ExecutionProviders& ep_providers,
                       const logging::Logger& logger,
                       std::optional<LayeringIndex>& layering_index);

  // Returns the Layering Rule indices mapped to the EP if any
  std::optional<std::reference_wrapper<const InlinedHashSet<size_t>>>
  GetLayeringRulesForThisEp(const std::string& ep_type) const;

  // Returns the parsed layering rules
  const LayeringRules& GetRules() const noexcept { return rules_; }

  // This function returns an index for the Layering rule the node is assigned to if any
  std::optional<size_t> GetNodeAssignment(const Graph& graph, NodeIndex node_id) const;

  // This is used when an EP fails to claim a node during partitioning so we make it
  // available for other EPs
  void MakeNodeUnassigned(const Graph& graph, NodeIndex node_id);
  /// <summary>
  /// Updates the layering index for a specific set of nodes in a graph.
  /// This checks if the nodes have annotations, and if so, matches them against the rules
  /// and updates the assignment.
  /// </summary>
  /// <param name="graph">The graph containing the nodes.</param>
  /// <param name="nodes">Indices of nodes to check and update.</param>
  void Update(const Graph& graph, gsl::span<const NodeIndex> nodes);

 private:
  LayeringRules rules_;
  LayeringRuleMatcher matcher_;
  // These stay constant
  EpNameToLayeringIndices ep_name_to_layering_indices_;
  LayeringIndexToEpName layering_index_to_ep_name_;

  using SetOfNodes = InlinedHashSet<NodeIndex>;
  using LayerIndexToNodes = InlinedHashMap<size_t, SetOfNodes>;
  using NodeIndexToLayeringIndex = InlinedHashMap<NodeIndex, size_t>;

  /// <summary>
  /// This struct contains the result of layering assignment for a graph.
  /// The struct first reflects pre-assignment according to the configuration.
  /// However, as we partition the graph, some nodes may be moved to unassigned sections
  /// to make them available to subsequent partitioning passes.
  /// </summary>
  struct GraphLayeringIndex {
    // Node to layering idx assignment map 1:1
    // If the node is not in this map, it is unassigned
    NodeIndexToLayeringIndex node_to_layering_index_;
    // This map contains mapping of LayeringRule index to the list of node ids
    // Reverse from the above 1:M
    LayerIndexToNodes layer_to_node_ids_;
  };

  LayeringIndex(LayeringRules layering_rules, EpNameToLayeringIndices ep_name_to_layering_indices, LayeringIndexToEpName layering_index_to_ep_name)
      : rules_(std::move(layering_rules)),
        matcher_(rules_),
        ep_name_to_layering_indices_(std::move(ep_name_to_layering_indices)),
        layering_index_to_ep_name_(std::move(layering_index_to_ep_name)) {}

  // Graph and sub-graphs mapping to their indices
  InlinedHashMap<const Graph*, GraphLayeringIndex> graph_index_;

  void ProcessGraph(const Graph& graph, std::optional<size_t> parent_layer_id);
};

}  // namespace onnxruntime

#else
namespace onnxruntime {
class LayeringIndex;
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
