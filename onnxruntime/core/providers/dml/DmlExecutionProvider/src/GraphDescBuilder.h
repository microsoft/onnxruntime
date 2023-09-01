// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "MLOperatorAuthorImpl.h"

namespace Dml
{
    struct GraphNodeProperties
    {
        std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfo> 
            internalRegInfo;

        // These are currently passed from the partitioning step since the only DML operators current 
        // supporting graph nodes don't customize the order of edges or shapes, other than coercing
        // dimension count.  This will change as the supported set of operators as graph nodes increases.
        Windows::AI::MachineLearning::Adapter::EdgeShapes inputShapes;
        Windows::AI::MachineLearning::Adapter::EdgeShapes outputShapes;
    };

    namespace GraphDescBuilder
    {
        // Gets a unique name for the node which survives recreation and graph manipulations between the point
        // that graph partitioning occurs and kernel creation happens
        const std::string& GetUniqueNodeName(const onnxruntime::Node& node);

        struct NodeInfo
        {
            Microsoft::WRL::ComPtr<IDMLOperator> op;
            std::string name;
        };

        struct GraphDesc
        {
            std::vector<NodeInfo> nodes;
            std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
            std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
            bool reuseCommandList;
        };

        GraphDesc BuildGraphDesc(
            const uint8_t* isConstGpuGraphInput,
            const size_t isConstGpuGraphInputCount,
            const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& isInitializerTransferable,
            const onnxruntime::Graph& graph,
            const onnxruntime::IndexedSubGraph& indexedSubGraph,
            const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
            IDMLDevice* device,
            const void* executionHandle);
    }
}