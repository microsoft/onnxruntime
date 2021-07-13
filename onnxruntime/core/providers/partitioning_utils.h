// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/graph/basic_types.h"

namespace onnxruntime {
struct ComputeCapability;
class GraphViewer;
class NodeArg;
class Node;

namespace utils {

using IsNodeSupportedFn = std::function<bool(const Node&)>;
using OnPartitionClosedFn = std::function<bool(const std::vector<const Node*>&)>;
using GenerateMetadefNameFn = std::function<std::string()>;

// TODO doc
std::vector<std::unique_ptr<ComputeCapability>>
CreateSupportedPartitions(const GraphViewer& graph_viewer,
                          const IsNodeSupportedFn& is_node_supported_fn,
                          const OnPartitionClosedFn& on_partition_closed_fn,
                          const GenerateMetadefNameFn& generate_metadef_name_fn,
                          const std::string& execution_provider_name,
                          bool debug_output = false);

/** 
Create the supported partitions for the execution provider. 

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param supported_nodes Set of nodes that the execution provider wants to handle. 
@param stop_ops Set of operator names at which we stop considering nodes for assignment to this execution provider.
@param generate_metadef_name Functor to create the name for the MetaDef. 
                             Most likely should call IExecutionProvider::GenerateMetaDefId. 
                             See onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.cc
                             for example usage.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.
@param debug_output Print diagnostic output about the partitions and reasons for partition breaks. 
                    No-op in a release build.

@returns ComputeCapability instances for all partitions assigned to the execution provider. 
*/
std::vector<std::unique_ptr<ComputeCapability>> CreateSupportedPartitions(
    const GraphViewer& graph_viewer,
    const std::unordered_set<const Node*>& supported_nodes,
    const std::unordered_set<std::string>& stop_ops,
    const GenerateMetadefNameFn& generate_metadef_name,
    const std::string& execution_provider_name,
    bool debug_output = false);

// TODO doc
std::unordered_set<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
                                                      const std::unordered_set<std::string>& stop_ops);

/**
Create a ComputeCapability instance from the group of nodes.
Will automatically determine the inputs and outputs required.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param group Group of nodes to include in the ComputeCapability instance.
@param generate_metadef_name Functor to create the name for the MetaDef.
                             Most likely should call IExecutionProvider::GenerateMetaDefId.
                             See onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.cc for
                             example usage.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.

@returns New ComputeCapability instance.

@remarks Prefer using CreateSupportedPartitions where possible, but if you need custom handling this provides a
         convenient way to correctly create the ComputeCapability instance.
*/
std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                         const std::vector<const Node*>& group,
                                                         const GenerateMetadefNameFn& generate_metadef_name,
                                                         const std::string& execution_provider_name);
}  // namespace utils
}  // namespace onnxruntime
