// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <unordered_map>
#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"

namespace Dml
{
class ExecutionProviderImpl;

class DmlGraphFusionTransformer : public onnxruntime::GraphTransformer
{
public:
    DmlGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider,
        const bool graphSerializationEnabled
    );

public:
    static inline const char* const DML_GRAPH_FUSION_NODE_NAME_PREFIX = "DmlFusedNode_";
    static inline const char* const DML_GRAPH_FUSION_NODE_DOMAIN = "DmlFusedNodeDomain";

private:
    onnxruntime::common::Status ApplyImpl(onnxruntime::Graph& graph,
                                            bool& modified,
                                            int graph_level,
                                            const onnxruntime::logging::Logger& logger) const final;

    onnxruntime::common::Status ApplyImplHelper(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const onnxruntime::logging::Logger& logger,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const;

private:
    const ExecutionProviderImpl* m_providerImpl = nullptr;
    const bool graphSerializationEnabled = false;
};
}
