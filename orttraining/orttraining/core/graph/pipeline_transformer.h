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
    Graph& graph,
    const std::unordered_set<std::string>& initializer_names_to_preserve,
    std::vector<std::string> graph_output_names,
    std::vector<ONNX_NAMESPACE::TensorShapeProto> graph_output_shapes,
    std::string& forward_recv_waited_event_name,
    std::string& forward_recv_wait_output_name,
    std::string& forward_recv_recorded_event_name,
    std::string& forward_recv_record_output_name,
    // Event ops' inputs and outputs related to forward Send.
    std::string& forward_send_waited_event_name,
    std::string& forward_send_wait_output_name,
    std::string& forward_send_recorded_event_name,
    std::string& forward_send_record_output_name,
    // Event ops' inputs and outputs related to backward Recv.
    std::string& backward_recv_waited_event_name,
    std::string& backward_recv_wait_output_name,
    std::string& backward_recv_recorded_event_name,
    std::string& backward_recv_record_output_name,
    // Event ops' inputs and outputs related to backward Send.
    std::string& backward_send_waited_event_name,
    std::string& backward_send_wait_output_name,
    std::string& backward_send_recorded_event_name,
    std::string& backward_send_record_output_name,
    // Event ops' inputs and outputs related to forward Compute.
    std::string& forward_compute_waited_event_name,
    std::string& forward_compute_wait_output_name,
    std::string& forward_compute_recorded_event_name,
    std::string& forward_compute_record_output_name,
    // Event ops' inputs and outputs related to backward Compute.
    std::string& backward_compute_waited_event_name,
    std::string& backward_compute_wait_output_name,
    std::string& backward_compute_recorded_event_name,
    std::string& backward_compute_record_output_name);

Status ApplyPipelinePartitionToMainGraph(
    Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cut_info,
    size_t pipeline_stage_id,
    size_t num_pipeline_stage);
}  // namespace training
}  // namespace onnxruntime
