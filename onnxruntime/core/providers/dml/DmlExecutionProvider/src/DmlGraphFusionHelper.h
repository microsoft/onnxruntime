#pragma once

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "ExecutionProvider.h"
#include "GraphPartitioner.h"
#include "FusedGraphKernel.h"
#include "MLOperatorAuthorImpl.h"


namespace Dml
{
namespace DmlGraphFusionHelper
{
    template <typename T>
    static T AlignToPow2(T offset, T alignment)
    {
        static_assert(std::is_unsigned_v<T>);
        assert(alignment != 0);
        assert((alignment & (alignment - 1)) == 0);
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateResource(
        const ExecutionProviderImpl* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize);

    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateCpuResource(
        const ExecutionProviderImpl* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize);

    void UnwrapTensor(
        Windows::AI::MachineLearning::Adapter::IWinmlExecutionProvider* winmlProvider,
        const onnxruntime::Tensor* tensor,
        ID3D12Resource** resource,
        uint64_t* allocId);

    std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>>
    GetInitializerToPartitionMap(
        const onnxruntime::GraphViewer& graph,
        gsl::span<std::unique_ptr<GraphPartition>> partitions
    );

    template <size_t AllocatorSize>
    void ConvertGraphDesc(
        const Dml::GraphDescBuilder::GraphDesc& graphDesc,
        const uint32_t inputCount,
        const uint32_t outputCount,
        IDMLDevice* device,
        StackAllocator<AllocatorSize>& allocator,
        const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToSubgraphInputIndex,
        const std::unordered_map<std::string_view, uint32_t>* serializedGraphLargeConstantNameToSubgraphInputIndex,
        _Out_ DML_GRAPH_DESC& dmlGraphDesc,
        _Inout_ std::vector<ComPtr<IDMLOperator>>& dmlOperators,
        _Inout_ std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges);

    onnxruntime::IndexedSubGraph CreateIndexedSubGraph(
        GraphPartition* partition,
        uint32_t partitionIndex,
        const std::string& partitionKernelPrefix);

    std::unordered_map<std::string, GraphNodeProperties> CreatePartitionNodePropsMap(
        const onnxruntime::Graph& graph,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>&& graphNodePropertyMap);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> TryCreateCompiledOperator(
        const GraphDescBuilder::GraphDesc& graphDesc,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        const ExecutionProviderImpl* providerImpl,
        const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToSubgraphInputIndex,
        const std::unordered_map<std::string_view, uint32_t>* serializedGraphLargeConstantNameToSubgraphInputIndex);

    void FusePartitionAndRegisterKernel(
        const uint32_t partitionIndex,
        onnxruntime::Graph& graph,
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& initializerNameToInitializerMap,
        const ExecutionProviderImpl* providerImpl,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        std::vector<uint8_t>&& isInputsUploadedByDmlEP,
        const GraphDescBuilder::GraphDesc& graphDesc,
        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator,
        const bool graphSerializationEnabled,
        const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToSubgraphInputIndex = nullptr,
        const std::unordered_map<std::string_view, uint32_t>* serializedGraphLargeConstantNameToSubgraphInputIndex = nullptr);

    void RegisterDynamicKernel(
        onnxruntime::Graph& graph,
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const ExecutionProviderImpl* providerImpl,
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap,
        const std::unordered_set<std::string>& dynamicCpuInputMap,
        std::shared_ptr<const onnxruntime::IndexedSubGraph> indexedSubGraph,
        std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>&& isInitializerTransferable);
}
}
