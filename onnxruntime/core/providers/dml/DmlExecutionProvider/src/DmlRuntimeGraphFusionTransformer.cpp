#pragma once

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "ExecutionProvider.h"
#include "DmlRuntimeGraphFusionTransformer.h"
#include "GraphPartitioner.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/kernel_lookup.h"
#include "core/optimizer/constant_sharing.h"
#include "DmlRuntimeFusedGraphKernel.h"
#include "MLOperatorAuthorImpl.h"
#include "DmlRuntimeGraphFusionHelper.h"


namespace Dml
{
    DmlRuntimeGraphFusionTransformer::DmlRuntimeGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider
    )
        :onnxruntime::GraphTransformer(name),
         m_providerImpl(static_cast<const ExecutionProvider*>(provider)->GetImpl())
    {
    }

    onnxruntime::common::Status DmlRuntimeGraphFusionTransformer::ApplyImpl(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const onnxruntime::logging::Logger& logger) const
    {
        return ApplyImplHelper(graph, modified, graph_level, logger, {});
    }

    onnxruntime::common::Status DmlRuntimeGraphFusionTransformer::ApplyImplHelper(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const onnxruntime::logging::Logger& logger,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const
    {
        onnxruntime::ProviderType provider_type = onnxruntime::kDmlExecutionProvider;
        const gsl::not_null<const onnxruntime::KernelRegistry*> registry = m_providerImpl->GetKernelRegistry().get();
        const auto kernel_type_str_resolver = onnxruntime::OpSchemaKernelTypeStrResolver{};
        const auto kernel_lookup = onnxruntime::KernelLookup{provider_type,
                                                             gsl::make_span(&registry, 1),
                                                             kernel_type_str_resolver};

        onnxruntime::GraphViewer graph_viewer(graph);
        const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

        for (auto node_index : node_topology_list)
        {
            auto* node = graph.GetNode(node_index);
            if (!node)
            {
                continue;  // node was removed
            }

            std::unordered_map<std::string, const onnxruntime::NodeArg*> subgraphImplicitInputDefs;
            for (const onnxruntime::NodeArg* inputDef : node->ImplicitInputDefs())
            {
                subgraphImplicitInputDefs[inputDef->Name()] = inputDef;
            }

            for (auto& entry : node->GetAttributeNameToMutableSubgraphMap())
            {
                auto& subgraph = *entry.second;
                ORT_RETURN_IF_ERROR(ApplyImplHelper(subgraph, modified, graph_level + 1, logger, subgraphImplicitInputDefs));
            }
        }

        // Initializers needed by any graph partition
        std::vector<onnxruntime::NodeIndex> additionalSplittingNodes;
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap;
        std::unordered_set<std::string> requiredInitializerMap;
        onnxruntime::GraphViewer graphViewer(graph);
        std::vector<std::unique_ptr<GraphPartition>> partitions = BuildPartitions(
            graphViewer,
            *m_providerImpl->GetInternalRegistrationInfoMap(),
            kernel_lookup,
            m_providerImpl->GetSupportedDeviceDataTypeMask(),
            graphNodePropertyMap,
            requiredInitializerMap,
            additionalSplittingNodes,
            implicitInputDefs,
            true);

        // Reset the splitting nodes for the current iteration
        additionalSplittingNodes.clear();

        // Reset the compiled operators for the current iteration
        std::vector<std::shared_ptr<onnxruntime::IndexedSubGraph>> indexedSubGraphs(partitions.size());

        // Create a map between each initialized tensor and the partition(s) it is part of.
        auto initializerPartitionMap = DmlRuntimeGraphFusionHelper::GetInitializerToPartitionMap(graphViewer, partitions);

        for (uint32_t partitionIndex = 0; partitionIndex < partitions.size(); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            if (partition->GetRootMergedPartition() != partition.get() ||
                !partition->IsDmlPartition())
            {
                continue;
            }

            if (partition->IsDmlGraphPartition())
            {
                std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> isInitializerTransferable;

                std::string partitionKernelPrefix = std::to_string(m_providerImpl->GetPartitionKernelPrefixVal()) + "_";
                m_providerImpl->IncreasePartitionKernelPrefixVal();

                indexedSubGraphs[partitionIndex] = std::make_shared<onnxruntime::IndexedSubGraph>(
                    DmlRuntimeGraphFusionHelper::CreateIndexedSubGraph(partition.get(), partitionIndex, partitionKernelPrefix));
            }
        }

        for (auto&& indexedSubGraph : indexedSubGraphs)
        {
            // Null compiled operators were not DML partitions
            if (indexedSubGraph)
            {
                DmlRuntimeGraphFusionHelper::RegisterKernel(
                    graph,
                    m_providerImpl->GetKernelRegistry().get(),
                    m_providerImpl,
                    graphNodePropertyMap,
                    std::move(indexedSubGraph));
            }
        }

        return onnxruntime::common::Status::OK();
    }
}
