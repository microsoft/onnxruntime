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
        _Out_ DML_GRAPH_DESC& dmlGraphDesc,
        IDMLDevice* device,
        StackAllocator<AllocatorSize>& allocator,
        std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
        std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
        std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
        std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges,
        std::vector<ComPtr<IDMLOperator>>& dmlOperators,
        const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToMainGraphInputIndex = nullptr,
        const std::unordered_map<std::string_view, uint32_t>* serializedGraphConstantNameToMainGraphInputIndex = nullptr);

    void CreateIDmlCompiledOperatorAndRegisterKernel(
        const uint32_t partitionIndex,
        onnxruntime::Graph& graph,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        const onnxruntime::Node& fusedNode,
        const std::unordered_map<std::string, GraphNodeProperties>& partitionNodePropsMap,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& isInitializerTransferable,
        const ExecutionProviderImpl* providerImpl,
        onnxruntime::KernelRegistry* registryForPartitionKernels);

    void FusePartitionAndRegisterKernel(
        GraphPartition* partition,
        uint32_t partitionIndex,
        onnxruntime::Graph& graph,
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>& graphNodePropertyMap,
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const std::string& partitionKernelPrefix,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& isInitializerTransferable,
        const ExecutionProviderImpl* providerImpl);
}
}
