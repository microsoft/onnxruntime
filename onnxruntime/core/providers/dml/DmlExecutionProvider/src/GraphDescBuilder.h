// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "MLOperatorAuthorImpl.h"
#include "ExecutionProvider.h"

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
        constexpr uint32_t minNodeCountToReuseCommandList = 5;
        constexpr uint32_t c_maxConstNodeDataSize = 8;

        // Gets a unique name for the node which survives recreation and graph manipulations between the point
        // that graph partitioning occurs and kernel creation happens
        const std::string& GetUniqueNodeName(const onnxruntime::Node& node);

        struct GraphDesc : DmlSerializedGraphDesc
        {
            bool reuseCommandList;
            Windows::AI::MachineLearning::Adapter::EdgeShapes outputShapes;
        };

        GraphDesc BuildGraphDesc(
            const uint8_t* isConstGpuGraphInput,
            const size_t isConstGpuGraphInputCount,
            const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& isInitializerTransferable,
            const std::unordered_map<std::string, GraphNodeProperties>& graphNodePropertyMap,
            const ExecutionProviderImpl* executionHandle,
            const onnxruntime::Path& modelPath,
            gsl::span<const onnxruntime::Node* const> subgraphNodes,
            gsl::span<const onnxruntime::NodeArg* const> subgraphInputs,
            gsl::span<const onnxruntime::NodeArg* const> subgraphOutputs,
            /*out*/ std::unordered_map<uint32_t, uint32_t>& serializedGraphInputIndexToSubgraphInputIndex,
            /*out*/ std::unordered_map<std::string_view, uint32_t>& serializedGraphLargeConstantNameToSubgraphInputIndex,
            /*out*/ std::vector<std::unique_ptr<std::byte[]>>& smallConstantData);
    }
}
