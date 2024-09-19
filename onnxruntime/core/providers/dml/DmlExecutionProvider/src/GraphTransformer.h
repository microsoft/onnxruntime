// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// Lotus framework headers for onnxruntime::IExecutionProvider (not part of the operator ABI).
#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"

namespace Dml
{
    class ExecutionProviderImpl;

    // Applies transforms to a Lotus graph. The graph transformer is responsible for setting the execution provider
    // on the graph nodes which DML supports.
    class GraphTransformer : public onnxruntime::GraphTransformer
    {
    public:
        GraphTransformer(
            const std::string& name,
            const onnxruntime::IExecutionProvider* provider
        );

    private:
     onnxruntime::common::Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const onnxruntime::logging::Logger& logger) const final;

    private:
        void PerformOperatorFusion(onnxruntime::Graph* graph, bool isMcdmDevice, bool* modified) const;

        const ExecutionProviderImpl* m_providerImpl = nullptr;
    };

} // namespace Dml
