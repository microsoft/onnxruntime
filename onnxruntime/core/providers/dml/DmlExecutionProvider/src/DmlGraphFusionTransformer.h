// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <unordered_map>
#include <wrl/client.h>
#include <d3d12.h>
#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"

namespace Dml
{
class ExecutionProviderImpl;

struct DmlGraphFusionCache
{
    std::vector<std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>> nonOwnedGraphInputsFromInitializers;
    std::unordered_map<std::string, std::unique_ptr<DmlGraphFusionCache>> subgraphCaches;
};

class DmlGraphFusionTransformer : public onnxruntime::GraphTransformer
{
public:
    DmlGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider,
        DmlGraphFusionCache& graph_fusion_cache
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
        DmlGraphFusionCache& cache,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const;

private:
    const ExecutionProviderImpl* m_providerImpl = nullptr;
    bool m_reuseWeights = false;
    DmlGraphFusionCache& m_graphFusionCache;
};
}
