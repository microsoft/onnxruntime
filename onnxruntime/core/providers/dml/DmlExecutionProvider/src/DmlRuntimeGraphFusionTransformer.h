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

class DmlRuntimeGraphFusionTransformer : public onnxruntime::GraphTransformer
{
public:
    DmlRuntimeGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider
    );

public:
    static inline const char* const DML_GRAPH_FUSION_NODE_NAME_PREFIX = "DmlRuntimeFusedNode_";
    static inline const char* const DML_GRAPH_FUSION_NODE_DOMAIN = "DmlRuntimeFusedNodeDomain";

private:
    onnxruntime::common::Status ApplyImpl(onnxruntime::Graph& graph,
                                          bool& modified,
                                          int graphLevel,
                                          const onnxruntime::logging::Logger& logger) const final;

    onnxruntime::common::Status ApplyImplHelper(
        onnxruntime::Graph& graph,
        bool& modified,
        int graphLevel,
        const onnxruntime::logging::Logger& logger,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const;

private:
    const ExecutionProviderImpl* m_providerImpl = nullptr;
};
}
