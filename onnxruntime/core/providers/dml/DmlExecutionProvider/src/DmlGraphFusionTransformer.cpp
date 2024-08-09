#pragma once

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "ExecutionProvider.h"
#include "DmlGraphFusionTransformer.h"
#include "GraphPartitioner.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/kernel_lookup.h"
#include "core/optimizer/constant_sharing.h"
#include "FusedGraphKernel.h"
#include "MLOperatorAuthorImpl.h"
#include "DmlGraphFusionHelper.h"


namespace Dml
{
    namespace
    {
        struct CompiledPartitionInfo
        {
            Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledOperator;
            onnxruntime::IndexedSubGraph indexedSubGraph;
            std::vector<uint8_t> isInputsUploadedByDmlEP;
            GraphDescBuilder::GraphDesc graphDesc;
            std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> isInitializerTransferable;
            std::vector<std::unique_ptr<std::byte[]>> smallConstantData; // Need to keep it alive for maintaining lifetime
            std::unordered_map<uint32_t, uint32_t> serializedGraphInputIndexToSubgraphInputIndex;
            std::unordered_map<std::string_view, uint32_t> serializedGraphLargeConstantNameToSubgraphInputIndex;
        };
    }

    DmlGraphFusionTransformer::DmlGraphFusionTransformer(
        const std::string& name,
        const onnxruntime::IExecutionProvider* provider,
        const bool graphSerializationEnabled
    )
        :onnxruntime::GraphTransformer(name),
         m_providerImpl(static_cast<const ExecutionProvider*>(provider)->GetImpl()),
         graphSerializationEnabled(graphSerializationEnabled)
    {
    }

    onnxruntime::common::Status DmlGraphFusionTransformer::ApplyImpl(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const onnxruntime::logging::Logger& logger) const
    {
        return ApplyImplHelper(graph, modified, graph_level, logger, {});
    }

    onnxruntime::common::Status DmlGraphFusionTransformer::ApplyImplHelper(
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

        std::vector<std::shared_ptr<CompiledPartitionInfo>> compiledPartitionInfos;
        std::vector<onnxruntime::NodeIndex> additionalSplittingNodes;

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

        do
        {
            // Initializers needed by any graph partition
            std::unordered_set<std::string> requiredInitializerMap;
            std::unordered_set<std::string> dynamicCpuInputMap;
            std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap;
            onnxruntime::GraphViewer graphViewer(graph);
            std::vector<std::unique_ptr<GraphPartition>> partitions = BuildPartitions(
                graphViewer,
                *m_providerImpl->GetInternalRegistrationInfoMap(),
                kernel_lookup,
                m_providerImpl->GetSupportedDeviceDataTypeMask(),
                graphNodePropertyMap,
                requiredInitializerMap,
                dynamicCpuInputMap,
                additionalSplittingNodes,
                implicitInputDefs,
                false);

            // Reset the splitting nodes for the current iteration
            additionalSplittingNodes.clear();

            // Reset the compiled operators for the current iteration
            compiledPartitionInfos.clear();
            compiledPartitionInfos.resize(partitions.size());

            // Create a map between each initialized tensor and the partition(s) it is part of.
            auto initializerPartitionMap = DmlGraphFusionHelper::GetInitializerToPartitionMap(graphViewer, partitions);

            for (uint32_t partitionIndex = 0; partitionIndex < partitions.size(); ++partitionIndex)
            {
                auto& partition = partitions[partitionIndex];

                if (partition->GetRootMergedPartition() != partition.get() ||
                    !partition->IsDmlPartition())
                {
                    continue;
                }

                // This map will tell which initializer can be removed from onnxruntime::Graph (and from it's field
                // onnx::GraphProto) while we upload the initializer to GPU.
                // Why we want to remove the initializer from ORT?
                //  1. To keep the peak memory usage as low as possible. That's why we are doing incremental upload to GPU.
                // What is initializer?
                //  An initializer is a input tensor to an operator or the graph itself, which is contant and will never change.
                // Why are we uploading the initialzer now?
                //  This prevents OnnxRuntime from allocating GPU resources and uploading those initializers,
                //  so the partiton's kernel can do so. In the process, it will pre-process weights while consuming a CPU
                //  backed resource, avoiding an extra set of GPU resources in memory.
                std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> isInitializerTransferable;

                if (partition->IsDmlGraphPartition())
                {
                    // populate isInitializerTransferable
                    for (const auto& input : partition->GetInputs())
                    {
                        const onnx::TensorProto* tensor = nullptr;
                        if (graph.GetInitializedTensor(input, tensor))
                        {
                            // It's only safe to transfer tensors which are used by this partition alone.
                            auto iter = initializerPartitionMap.find(tensor);
                            assert(iter != initializerPartitionMap.end());
                            if (iter->second.size() > 1)
                            {
                                // By including non-transferrable tensors in isInitializerTransferable, it causes DML to upload and preprocess them
                                // to duplicate locations rather than treating them as being non-constant, which is helpful for optimization.
                                // The size threshold for this should be no smaller than that used to combine initializers in the constant
                                // sharing transform to prevent that transform from hurting performance.
                                // If the kernel relies on this input to be initialized, it should also be small enough to copy cheaply.
                                constexpr uint64_t maximumElementsForDuplicationTensor = 64;
                                static_assert(maximumElementsForDuplicationTensor >= onnxruntime::ConstantSharing::TENSOR_ELEM_COUNT_THRESHOLD);

                                uint64_t totalElementCount = 1;
                                for (int i = 0; i < tensor->dims().size(); ++i)
                                {
                                    totalElementCount *= tensor->dims()[i];
                                }

                                if (totalElementCount <=  maximumElementsForDuplicationTensor ||
                                    requiredInitializerMap.find(input) != requiredInitializerMap.end())
                                {
                                    isInitializerTransferable[input] = {tensor, false};
                                }

                                continue;
                            }
                            isInitializerTransferable[input] = {tensor, true};
                        }
                    }

                    std::string partitionKernelPrefix = std::to_string(m_providerImpl->GetPartitionKernelPrefixVal()) + "_";
                    m_providerImpl->IncreasePartitionKernelPrefixVal();

                    auto indexedSubGraph = DmlGraphFusionHelper::CreateIndexedSubGraph(partition.get(), partitionIndex, partitionKernelPrefix);

                    // Create a map of which inputs are uploaded by the DML EP
                    const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(indexedSubGraph.GetMetaDef()->inputs.size());
                    std::vector<uint8_t> isInputsUploadedByDmlEP(fusedNodeInputCount);
                    for (uint32_t index = 0; index < fusedNodeInputCount; ++index)
                    {
                        auto iter = isInitializerTransferable.find(indexedSubGraph.GetMetaDef()->inputs[index]);
                        isInputsUploadedByDmlEP[index] = iter != isInitializerTransferable.end() ? true : false;
                    }

                    auto partitionNodePropsMap = DmlGraphFusionHelper::CreatePartitionNodePropsMap(
                        graph,
                        indexedSubGraph,
                        std::move(graphNodePropertyMap));

                    // Convert partitionONNXGraph into DML EP GraphDesc
                    auto modelPath = graph.ModelPath();

                    const gsl::span<const std::string> subGraphInputArgNames = indexedSubGraph.GetMetaDef()->inputs;
                    const gsl::span<const std::string> subGraphOutputArgNames = indexedSubGraph.GetMetaDef()->outputs;

                    std::vector<const onnxruntime::Node*> subgraphNodes;
                    subgraphNodes.reserve(indexedSubGraph.nodes.size());

                    std::vector<const onnxruntime::NodeArg*> subgraphInputs;
                    subgraphInputs.reserve(subGraphInputArgNames.size());

                    std::vector<const onnxruntime::NodeArg*> subgraphOutputs;
                    subgraphOutputs.reserve(subGraphOutputArgNames.size());

                    for (size_t sortedNodeIndex : indexedSubGraph.nodes)
                    {
                        subgraphNodes.push_back(graph.GetNode(sortedNodeIndex));
                    }

                    for (const std::string& graphInputName : subGraphInputArgNames)
                    {
                        subgraphInputs.push_back(graph.GetNodeArg(graphInputName));
                    }

                    for (const std::string& graphOutputName : subGraphOutputArgNames)
                    {
                        subgraphOutputs.push_back(graph.GetNodeArg(graphOutputName));
                    }

                    ComPtr<IDMLDevice> device;
                    ORT_THROW_IF_FAILED(m_providerImpl->GetDmlDevice(device.GetAddressOf()));
                    // This map will be used to transfer the initializer to D3D12 system heap memory.
                    // 'serializedDmlGraphDesc' will have constant input as intermediate edges, that's why
                    // we need a mapping between intermediateEdgeIndex and indexedSubGraph's (a given partition)
                    // input arg index.
                    //   For ex: Let's say intermediate edge index = idx, then
                    //           indexedSubGraphInputArgIdx = constantEdgeIdxToSubgraphInputArgIdxMap[idx];
                    //           corresponding constant tensor = initializerNameToInitializerMap[indexedSubGraph.GetMetaDef()->inputs[indexedSubGraphInputArgIdx]]
                    // We are using intermediate edge index as a key because same constant tensor can be used by
                    // multiple nodes.
                    std::unordered_map<uint32_t, uint32_t> serializedGraphInputIndexToSubgraphInputIndex;
                    std::unordered_map<std::string_view, uint32_t> serializedGraphLargeConstantNameToSubgraphInputIndex;
                    std::vector<std::unique_ptr<std::byte[]>> smallConstantData;
                    GraphDescBuilder::GraphDesc graphDesc = GraphDescBuilder::BuildGraphDesc(
                        isInputsUploadedByDmlEP.data(),
                        isInputsUploadedByDmlEP.size(),
                        isInitializerTransferable,
                        partitionNodePropsMap,
                        m_providerImpl,
                        modelPath,
                        subgraphNodes,
                        subgraphInputs,
                        subgraphOutputs,
                        serializedGraphInputIndexToSubgraphInputIndex,
                        serializedGraphLargeConstantNameToSubgraphInputIndex,
                        smallConstantData);

                    // Compile the operator
                    auto compiledPartition = DmlGraphFusionHelper::TryCreateCompiledOperator(
                        graphDesc,
                        indexedSubGraph,
                        m_providerImpl,
                        &serializedGraphInputIndexToSubgraphInputIndex,
                        &serializedGraphLargeConstantNameToSubgraphInputIndex);

                    if (!compiledPartition)
                    {
                        // Fail early if even a single operator is too big to compile. This is highly unlikely.
                        ORT_THROW_HR_IF(E_INVALIDARG, indexedSubGraph.nodes.size() < 2);

                        // Tell the partitioner to split the current partition in half, in the middle
                        additionalSplittingNodes.push_back(indexedSubGraph.nodes[indexedSubGraph.nodes.size() / 2]);

                        // Exit early since we need to repartition
                        break;
                    }
                    else
                    {
                        auto compiledPartitionInfo = std::make_shared<CompiledPartitionInfo>();
                        compiledPartitionInfo->compiledOperator = std::move(compiledPartition);
                        compiledPartitionInfo->indexedSubGraph = std::move(indexedSubGraph);
                        compiledPartitionInfo->isInputsUploadedByDmlEP = std::move(isInputsUploadedByDmlEP);
                        compiledPartitionInfo->graphDesc = std::move(graphDesc);
                        compiledPartitionInfo->isInitializerTransferable = std::move(isInitializerTransferable);
                        compiledPartitionInfo->smallConstantData = std::move(smallConstantData);
                        compiledPartitionInfo->serializedGraphInputIndexToSubgraphInputIndex = std::move(serializedGraphInputIndexToSubgraphInputIndex);
                        compiledPartitionInfo->serializedGraphLargeConstantNameToSubgraphInputIndex = std::move(serializedGraphLargeConstantNameToSubgraphInputIndex);
                        compiledPartitionInfos[partitionIndex] = std::move(compiledPartitionInfo);
                    }
                }
            }
        }
        while (!additionalSplittingNodes.empty());

        uint32_t partitionIndex = 0;
        for (auto&& compiledPartitionInfo : compiledPartitionInfos)
        {
            // Null compiled operators were not DML partitions
            if (compiledPartitionInfo)
            {
                DmlGraphFusionHelper::FusePartitionAndRegisterKernel(
                    partitionIndex++,
                    graph,
                    m_providerImpl->GetKernelRegistry().get(),
                    compiledPartitionInfo->isInitializerTransferable,
                    m_providerImpl,
                    compiledPartitionInfo->indexedSubGraph,
                    std::move(compiledPartitionInfo->isInputsUploadedByDmlEP),
                    compiledPartitionInfo->graphDesc,
                    compiledPartitionInfo->compiledOperator,
                    graphSerializationEnabled,
                    &compiledPartitionInfo->serializedGraphInputIndexToSubgraphInputIndex,
                    &compiledPartitionInfo->serializedGraphLargeConstantNameToSubgraphInputIndex);
            }
        }

        return onnxruntime::common::Status::OK();
    }
}
