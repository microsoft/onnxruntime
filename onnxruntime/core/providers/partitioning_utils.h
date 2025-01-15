// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// QDQ models require graph modification at runtime, so we know this infrastructure is not used in a minimal build
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
struct ComputeCapability;
class GraphViewer;
class Node;
class NodeArg;
class NodeUnit;

namespace utils {

/**
Called to check whether a node is supported.

@param node The node to check.

@return Whether the node is supported.
*/
using IsNodeSupportedFn = std::function<bool(const Node& node)>;

/**
Called to indicate a completed partition node group.
The partition is kept or discarded based on the return value.

@param group The partition node group.

@return Whether to keep the partition.
*/
using OnGroupClosedFn = std::function<bool(const std::vector<const Node*>& group)>;

/**
Called to create a metadef name.
Most likely should call ModelMetadefIdGenerator.GenerateId.
See onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.cc for example usage.

@return The metadef name.
*/
using GenerateMetadefNameFn = std::function<std::string()>;

/**
Create the supported partitions for the execution provider.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param is_node_supported_fn Callback to check whether a node is supported.
@param on_group_closed_fn Callback to indicate a completed partition node group.
@param generate_metadef_name_fn Callback to create the name for the MetaDef.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.
@param execution_provider_type ExecutionProviderType of the EP creating this ComputeCapability instance.
@param node_unit_map Map of each Node in the graph_viewer to its NodeUnit. Provide if EP handles QDQ format models.
                     Should be created by EP calling GetAllNodeUnits.
@param drop_constant_initializer Drop constant initializers from input to a ComputeCapability.
                                 Set to true if constant initializers have been copied into a compiled model to allow
                                 ORT to free the initializer. If the initializer remains as an input it will appear to
                                 still be in-use.
@returns ComputeCapability instances for all partitions assigned to the execution provider.
*/
std::vector<std::unique_ptr<ComputeCapability>>
CreateSupportedPartitions(const GraphViewer& graph_viewer,
                          const IsNodeSupportedFn& is_node_supported_fn,
                          const OnGroupClosedFn& on_group_closed_fn,
                          const GenerateMetadefNameFn& generate_metadef_name_fn,
                          const std::string& execution_provider_name,
                          const std::string& execution_provider_type,
                          const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map,
                          bool drop_constant_initializers = false);

/**
Create the supported partitions for the execution provider.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param supported_nodes Set of nodes that the execution provider wants to handle.
@param stop_ops Set of operator names at which we stop considering nodes for assignment to this execution provider.
@param generate_metadef_name Functor to create the name for the MetaDef.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.
@param execution_provider_type ExecutionProviderType of the EP creating this ComputeCapability instance.
@param node_unit_map Map of each Node in the graph_viewer to its NodeUnit. Provide if EP handles QDQ format models.
                     Should be created by EP calling GetAllNodeUnits.
@param drop_constant_initializer Drop constant initializers from input to a ComputeCapability.
                                 Set to true if constant initializers have been copied into a compiled model to allow
                                 ORT to free the initializer. If the initializer remains as an input it will appear to
                                 still be in-use.
@returns ComputeCapability instances for all partitions assigned to the execution provider.
*/
std::vector<std::unique_ptr<ComputeCapability>>
CreateSupportedPartitions(const GraphViewer& graph_viewer,
                          const std::unordered_set<const Node*>& supported_nodes,
                          const std::unordered_set<std::string>& stop_ops,
                          const GenerateMetadefNameFn& generate_metadef_name,
                          const std::string& execution_provider_name,
                          const std::string& execution_provider_type,
                          const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map,
                          bool drop_constant_initializers = false);

/**
Create a ComputeCapability instance from the group of nodes.
Will automatically determine the inputs and outputs required.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param group Group of nodes to include in the ComputeCapability instance.
@param generate_metadef_name Functor to create the name for the MetaDef.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.

@returns New ComputeCapability instance.

@remarks Prefer using CreateSupportedPartitions where possible, but if you need custom handling this provides a
         convenient way to correctly create the ComputeCapability instance.
*/
std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                         const std::vector<const Node*>& group,
                                                         const GenerateMetadefNameFn& generate_metadef_name,
                                                         const std::string& execution_provider_name,
                                                         bool drop_constant_initializers);

/**
Create the set of nodes to exclude based on a set of stop ops.
Stop op nodes and nodes downstream from them will be excluded.

@param graph_viewer GraphViewer with the nodes to consider.
@param stop_ops The set of stop ops.

@return The set of excluded nodes.
*/
InlinedHashSet<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
                                                  const std::unordered_set<std::string>& stop_ops);

/**
Create partition node groups.

A partition node group (a.k.a. a group) contains supported nodes that will run in a partition.

All nodes in a group can be run together. This means that two nodes with an intervening unsupported node cannot be in
the same group. On the other hand, nodes within the same group do not necessarily have to be connected.

The partitioning algorithm attempts to form the largest possible groups in a greedy fashion. It is a variant of Kahn's
topological sort algorithm that forms the group(s) as it goes.

Conceptually, we consider nodes in a sequence of waves starting from the root nodes. One wave produces at most one
group. A wave flows over nodes in topological order, adding supported nodes to the current group, and stops at the
border of the current group. The next wave starts where the previous wave stopped.

When generating the topological ordering, we maintain a set of nodes that have no inputs produced by unprocessed nodes.
From this set, we select the next node to process.

When selecting the next node to process, we first take:
- a supported node (which will be part of the group)
- an unsupported node that does not consume an output of any node in the group

The remaining unsupported nodes mark the border of the current group so they will be processed later when we consider
the next group.

If node_unit_map is provided, we process NodeUnit instances (a logical 'Node' that can be a single node or a
QDQ node group) instead of individual Node instances. As an EP must take complete NodeUnit instances (i.e. it
must not break up a QDQ node group by taking a subset of nodes in it), this granularity of processing is valid.
It is required to ensure we do not break up a QDQ node unit during partitioning.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param is_node_supported_fn Callback to check whether a node is supported.
@param on_group_closed_fn Callback to indicate a completed partition node group.
@return The partition node groups.
*/
std::vector<std::vector<const Node*>> CreateSupportedPartitionNodeGroups(
    const GraphViewer& graph_viewer,
    const IsNodeSupportedFn& is_node_supported_fn,
    const OnGroupClosedFn& on_group_closed_fn,
    const std::string& execution_provider_type,
    const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map);
}  // namespace utils
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
