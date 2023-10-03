// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "MLOperatorAuthorImpl.h"
#include "DmlRuntimeFusedGraphKernel.h"
#include "DmlRuntimeGraphFusionHelper.h"

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml
{
    class DmlRuntimeFusedGraphKernel : public onnxruntime::OpKernel
    {
    public:
        DmlRuntimeFusedGraphKernel() = delete;

        DmlRuntimeFusedGraphKernel(
            const onnxruntime::OpKernelInfo& kernelInfo,
            std::shared_ptr<const onnxruntime::IndexedSubGraph> indexedSubGraph,
            const onnxruntime::Path& modelPath,
            std::shared_ptr<std::vector<std::vector<std::string>>> inputDimParams,
            std::vector<std::shared_ptr<onnxruntime::Node>>&& subgraphNodes,
            std::vector<const onnxruntime::NodeArg*>&& subgraphInputs,
            std::vector<const onnxruntime::NodeArg*>&& subgraphOutputs,
            std::vector<std::shared_ptr<onnxruntime::NodeArg>>&& intermediateNodeArgs,
            std::unordered_map<std::string, GraphNodeProperties>&& partitionNodePropsMap,
            std::vector<ONNX_NAMESPACE::TensorProto>&& ownedInitializers)
        : OpKernel(kernelInfo),
          m_indexedSubGraph(std::move(indexedSubGraph)),
          m_modelPath(modelPath),
          m_inputDimParams(std::move(inputDimParams)),
          m_subgraphNodes(std::move(subgraphNodes)),
          m_subgraphInputs(std::move(subgraphInputs)),
          m_subgraphOutputs(std::move(subgraphOutputs)),
          m_intermediateNodeArgs(std::move(intermediateNodeArgs)),
          m_partitionNodePropsMap(std::move(partitionNodePropsMap)),
          m_ownedInitializers(std::move(ownedInitializers))
        {
            for (const auto& initializer : m_ownedInitializers)
            {
                m_isInitializerTransferable[initializer.name()] = std::make_pair(&initializer, false);
            }

            // Get the execution provider interfaces
            auto executionHandle = kernelInfo.GetExecutionProvider()->GetExecutionHandle();
            if (executionHandle)
            {
                // We assume the execution object inherits IUnknown as its first base
                ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));

                // Get the WinML-specific execution provider interface from the execution object.
                ORT_THROW_IF_FAILED(providerExecutionObject.As(&m_provider));
                ORT_THROW_IF_FAILED(providerExecutionObject.As(&m_winmlProvider));
            }

            m_subgraphNodePointers.reserve(m_subgraphNodes.size());

            for (auto& subgraphNode : m_subgraphNodes)
            {
                m_subgraphNodePointers.push_back(subgraphNode.get());
            }
        }

        void TranslateAndCompileGraph(
            const onnxruntime::OpKernelInfo& kernelInfo,
            std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
            std::vector<DML_BUFFER_BINDING> initInputBindings) const
        {
            std::optional<DML_BUFFER_BINDING> persistentResourceBinding;

            // Allocate a persistent resource and initialize the operator
            UINT64 persistentResourceSize = m_compiledExecutionPlanOperator->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_provider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Disabled,
                    m_persistentResource.GetAddressOf(),
                    m_persistentResourceAllocatorUnk.GetAddressOf()));

                persistentResourceBinding = DML_BUFFER_BINDING { m_persistentResource.Get(), 0, persistentResourceSize };
            }

            ORT_THROW_IF_FAILED(m_provider->InitializeOperator(
                m_compiledExecutionPlanOperator.Get(),
                persistentResourceBinding ? &*persistentResourceBinding : nullptr,
                gsl::make_span(initInputBindings)));

            // Queue references to objects which must be kept alive until resulting GPU work completes
            m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
            m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnk.Get());

            std::for_each(
                initializeResourceRefs.begin(),
                initializeResourceRefs.end(),
                [&](ComPtr<ID3D12Resource>& resource){ m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(resource).Get()); }
            );
        }

        onnxruntime::Status Compute(onnxruntime::OpKernelContext* kernelContext) const override
        {
            const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(m_indexedSubGraph->GetMetaDef()->inputs.size());

            // Populate input bindings for operator initialization
            std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> initializeResourceRefs; // For lifetime control
            std::vector<DML_BUFFER_BINDING> initInputBindings(fusedNodeInputCount);
            std::vector<uint8_t> isInputsUploadedByDmlEP(fusedNodeInputCount);

            auto providerImpl = static_cast<const ExecutionProvider*>(Info().GetExecutionProvider())->GetImpl();

            std::unordered_map<std::string, int64_t> dynamicDimOverrides;

            ORT_THROW_HR_IF(E_UNEXPECTED, m_inputDimParams->size() != kernelContext->InputCount());
            for (int inputIndex = 0; inputIndex < m_inputDimParams->size(); ++inputIndex)
            {
                const auto& input = kernelContext->RequiredInput<onnxruntime::Tensor>(inputIndex);
                ORT_THROW_HR_IF(E_UNEXPECTED, input.Shape().NumDimensions() != (*m_inputDimParams)[inputIndex].size());

                for (int i = 0; i < input.Shape().NumDimensions(); ++i)
                {
                    const std::string& dimParam = (*m_inputDimParams)[inputIndex][i];

                    if (!dimParam.empty())
                    {
                        dynamicDimOverrides[dimParam] = input.Shape().GetDims()[i];
                    }
                }
            }

            for (auto& subgraphNode : m_subgraphNodes)
            {
                for (onnxruntime::NodeArg* inputDef : subgraphNode->MutableInputDefs())
                {
                    ORT_THROW_HR_IF(E_INVALIDARG, !inputDef->TypeAsProto()->has_tensor_type());
                    auto tensorShape = inputDef->TypeAsProto()->tensor_type().shape();

                    for (int i = 0; i < tensorShape.dim_size(); ++i)
                    {
                        if (tensorShape.dim(i).has_dim_param())
                        {
                            tensorShape.mutable_dim(i)->set_dim_value(dynamicDimOverrides[tensorShape.dim(i).dim_param()]);
                        }
                    }

                    inputDef->SetShape(tensorShape);
                }
            }

            // Convert partitionONNXGraph into DML EP GraphDesc
            ComPtr<IDMLDevice> device;
            ORT_THROW_IF_FAILED(providerImpl->GetDmlDevice(device.GetAddressOf()));
            GraphDescBuilder::GraphDesc graphDesc = GraphDescBuilder::BuildGraphDesc(
                isInputsUploadedByDmlEP.data(),
                isInputsUploadedByDmlEP.size(),
                m_isInitializerTransferable,
                m_partitionNodePropsMap,
                device.Get(),
                providerImpl,
                m_modelPath,
                m_subgraphNodePointers,
                m_subgraphInputs,
                m_subgraphOutputs);

            // Walk through each graph edge and mark used inputs
            m_inputsUsed.resize(fusedNodeInputCount, false);
            for (const DML_INPUT_GRAPH_EDGE_DESC& edge : graphDesc.inputEdges)
            {
                m_inputsUsed[edge.GraphInputIndex] = true;
            }

            // Compile the operator
            m_compiledExecutionPlanOperator = DmlRuntimeGraphFusionHelper::TryCreateCompiledOperator(
                graphDesc,
                *m_indexedSubGraph,
                providerImpl);

            TranslateAndCompileGraph(
                Info(),
                initializeResourceRefs,
                initInputBindings);

            // Wrap tensors as required by Dml::IExecutionProvider::ExecuteOperator
            OpKernelContextWrapper contextWrapper(
                kernelContext,
                Info().GetExecutionProvider(),
                true,
                nullptr);

            ORT_THROW_IF_FAILED(m_provider->AddUAVBarrier());

            // Get input resources for execution, excluding those which were specified as owned by DML and provided
            // at initialization instead.
            std::vector<ComPtr<IMLOperatorTensor>> inputTensors(kernelContext->InputCount());
            std::vector<ID3D12Resource*> inputPtrs(kernelContext->InputCount());

            for (int i = 0; i < kernelContext->InputCount(); ++i)
            {
                if (!m_inputsUsed[i])
                {
                    continue;
                }

                ORT_THROW_IF_FAILED(contextWrapper.GetInputTensor(i, inputTensors[i].GetAddressOf()));
                inputPtrs[i] = m_provider->DecodeResource(MLOperatorTensor(inputTensors[i].Get()).GetDataInterface().Get());
            }

            auto aux = contextWrapper.GetOutputTensors(graphDesc.outputShapes);
            ExecuteOperator(
                m_compiledExecutionPlanOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                inputPtrs,
                aux);

            ORT_THROW_IF_FAILED(m_provider->AddUAVBarrier());

            // Queue references to objects which must be kept alive until resulting GPU work completes
            m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
            m_winmlProvider->QueueReference(m_persistentResourceAllocatorUnk.Get());

            return onnxruntime::Status::OK();
        }

        void ExecuteOperator(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<ID3D12Resource*> inputTensors,
            gsl::span<IMLOperatorTensor*> outputTensors) const
        {
            auto FillBindingsFromTensors = [this](auto& bufferBindings, auto& bindingDescs,  gsl::span<IMLOperatorTensor*>& tensors)
            {
                for (IMLOperatorTensor* tensor : tensors)
                {
                    if (tensor)
                    {
                        assert(tensor->IsDataInterface());
                        ID3D12Resource* resource = m_provider->DecodeResource(MLOperatorTensor(tensor).GetDataInterface().Get());
                        D3D12_RESOURCE_DESC resourceDesc = resource->GetDesc();
                        bufferBindings.push_back({ resource, 0, resourceDesc.Width });
                        bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                    }
                    else
                    {
                        bufferBindings.push_back({ nullptr, 0, 0 });
                        bindingDescs.push_back({ DML_BINDING_TYPE_NONE, nullptr });
                    }
                }
            };

            auto FillBindingsFromBuffers = [](auto& bufferBindings, auto& bindingDescs,  gsl::span<ID3D12Resource*>& resources)
            {
                for (ID3D12Resource* resource : resources)
                {
                    if (resource)
                    {
                        D3D12_RESOURCE_DESC resourceDesc = resource->GetDesc();
                        bufferBindings.push_back({ resource, 0, resourceDesc.Width });
                        bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                    }
                    else
                    {
                        bufferBindings.push_back({ nullptr, 0, 0 });
                        bindingDescs.push_back({ DML_BINDING_TYPE_NONE, nullptr });
                    }
                }
            };

            std::vector<DML_BUFFER_BINDING> inputBufferBindings;
            inputBufferBindings.reserve(inputTensors.size());
            std::vector<DML_BINDING_DESC> inputBindings;
            inputBindings.reserve(inputTensors.size());
            FillBindingsFromBuffers(inputBufferBindings, inputBindings, inputTensors);

            std::vector<DML_BUFFER_BINDING> outputBufferBindings;
            outputBufferBindings.reserve(outputTensors.size());
            std::vector<DML_BINDING_DESC> outputBindings;
            outputBindings.reserve(outputTensors.size());
            FillBindingsFromTensors(outputBufferBindings, outputBindings, outputTensors);

            ORT_THROW_IF_FAILED(m_provider->ExecuteOperator(
                op,
                persistentResourceBinding,
                inputBindings,
                outputBindings));
            }

    private:
        ComPtr<IWinmlExecutionProvider> m_winmlProvider;
        ComPtr<Dml::IExecutionProvider> m_provider;

        // Re-usable command list, supporting descriptor heap, and DML binding table to update that heap.
        ComPtr<ID3D12GraphicsCommandList> m_graphicsCommandList;
        ComPtr<ID3D12CommandAllocator> m_commandAllocator;
        ComPtr<ID3D12DescriptorHeap> m_heap;
        ComPtr<IDMLBindingTable> m_bindingTable;
        std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
        std::shared_ptr<const onnxruntime::IndexedSubGraph> m_indexedSubGraph;
        const onnxruntime::Path& m_modelPath;
        std::shared_ptr<std::vector<std::vector<std::string>>> m_inputDimParams;
        std::vector<std::shared_ptr<onnxruntime::Node>> m_subgraphNodes;
        std::vector<const onnxruntime::NodeArg*> m_subgraphInputs;
        std::vector<const onnxruntime::NodeArg*> m_subgraphOutputs;
        std::vector<std::shared_ptr<onnxruntime::NodeArg>> m_intermediateNodeArgs;
        std::unordered_map<std::string, GraphNodeProperties> m_partitionNodePropsMap;
        std::vector<ONNX_NAMESPACE::TensorProto> m_ownedInitializers;
        std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> m_isInitializerTransferable;
        std::vector<const onnxruntime::Node*> m_subgraphNodePointers;

        // Bindings from previous executions of a re-used command list
        mutable ComPtr<IDMLCompiledOperator> m_compiledExecutionPlanOperator;
        mutable std::vector<uint64_t> m_inputBindingAllocIds;
        mutable std::vector<uint64_t> m_outputBindingAllocIds;
        mutable uint64_t m_tempBindingAllocId = 0;
        mutable std::vector<bool> m_inputsUsed;
        mutable ComPtr<ID3D12Resource> m_persistentResource;
        mutable ComPtr<IUnknown> m_persistentResourceAllocatorUnk; // Controls when the persistent resource is returned to the allocator

        // Fence tracking the status of the command list's last execution, and whether its descriptor heap
        // can safely be updated.
        mutable ComPtr<ID3D12Fence> m_fence;
        mutable uint64_t m_completionValue = 0;
    };

    onnxruntime::OpKernel* CreateRuntimeFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        std::shared_ptr<const onnxruntime::IndexedSubGraph> indexedSubGraph,
        const onnxruntime::Path& modelPath,
        std::shared_ptr<std::vector<std::vector<std::string>>> inputDimParams,
        std::vector<std::shared_ptr<onnxruntime::Node>>&& subgraphNodes,
        std::vector<const onnxruntime::NodeArg*>&& subgraphInputs,
        std::vector<const onnxruntime::NodeArg*>&& subgraphOutputs,
        std::vector<std::shared_ptr<onnxruntime::NodeArg>>&& intermediateNodeArgs,
        std::unordered_map<std::string, GraphNodeProperties>&& partitionNodePropsMap,
        std::vector<ONNX_NAMESPACE::TensorProto>&& ownedInitializers)
    {
        return new DmlRuntimeFusedGraphKernel(
            info,
            std::move(indexedSubGraph),
            modelPath,
            std::move(inputDimParams),
            std::move(subgraphNodes),
            std::move(subgraphInputs),
            std::move(subgraphOutputs),
            std::move(intermediateNodeArgs),
            std::move(partitionNodePropsMap),
            std::move(ownedInitializers)
        );
    }
} // namespace Dml
