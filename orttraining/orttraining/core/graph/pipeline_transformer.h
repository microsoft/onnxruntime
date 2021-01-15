// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& node_arg_name);
void GetPipelineRecvInput(const Graph& graph, std::string& node_arg_name);

Status TransformGraphForPipeline(
    const bool keep_original_output_schema,
    const std::unordered_set<std::string>& weights_to_train,
    const std::unordered_map<std::string, std::vector<int>>& sliced_schema,
    const std::vector<std::string>& expected_output_names,
    Graph& graph,
    pipeline::PipelineTensorNames& pipeline_tensor_names);

// Partitions the graph into num_stages subgraphs, as defined in op_to_stage map,
// which maps operators to stage ids. After the graph is partitioned, it drops
// all the tensors and operators that do not belong to the subgraph pipeline_stage_id.
// TODO(jufranc): when adapting this code to partition training graphs, add
// a boolean is_training as parameter.
Status ApplyPipelinePartitionToMainGraph(Graph& graph,
                                         const std::map<const Node*, int>& op_to_stage,
                                         const int pipeline_stage_id,
                                         const int num_stages,
                                         const std::vector<int32_t>& rank_ids);

// First of two functions to obtain a mapping between operators and stage ids.
// Input:
//   - graph is the graph being partitioned into multiple pipeline stages.
//   - id_to_stage maps string identifiers of operators and stage ids. Each
// operator is identified with the name of any of its outputs.
//   - op_to_stage keeps the output of this function, where op_to_stage[node_ptr]
// is the pipeline stage ID of the pointed node.
//   - num_stages is the total number of stages.
Status GetDeviceAssignmentMap(const Graph& graph,
                              const std::map<std::string, int>& id_to_stage,
                              std::map<const Node*, int>& op_to_stage,
                              const int num_stages);

// Second of two functions to obtain a mapping between operators and stage ids.
// This version in particular converts a list of graph cuts (i.e., CutInfo)
// into a mapping between operators and stages.
// Input:
//   - graph is the graph being partitioned into multiple pipeline stages.
//   - cuts describes all the cut points as defined by the user (c.f., CutInfo
// type definition.
//   - op_to_stage keeps the output of this function, where op_to_stage[node_ptr]
// is the pipeline stage ID of the pointed node.
//   - num_stages is the total number of stages.
Status GetDeviceAssignmentMap(const Graph& graph,
                              const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cuts,
                              std::map<const Node*, int>& op_to_stage,
                              const int num_stages);

// This function creates data-dependency from "dependent_node_args"
// to "node". That is, it makes sure "node" is executed after
// the generation of "dependent_node_args."
// Assume that we have two independent sub-graphs
//
//  X -> ReLU1 -> Y
//  S -> ReLU2 -> T
//
// If we want to execute ReLU2 after ReLU1, we can do
//
//  X -> ReLU1 -> Y ------.
//                        |
//                        v
//                S -> PassThrough -> Y'
//                        |
//                        `---------> S' -> ReLU2 -> T
//
// In this case, "dependent_node_args" is "Y" and
// "node" is "ReLU1".
void SetDataDependency(
    Graph& graph,
    Node& postponed_node,
    const std::vector<NodeArg*>& dependent_node_args);

}  // namespace training
}  // namespace onnxruntime
