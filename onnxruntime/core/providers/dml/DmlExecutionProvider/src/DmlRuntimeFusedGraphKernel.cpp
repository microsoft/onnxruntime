// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlRuntimeFusedGraphKernel.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlGraphFusionHelper.h"

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
            std::vector<std::shared_ptr<onnxruntime::Node>>&& subgraphNodes,
            std::vector<const onnxruntime::NodeArg*>&& subgraphInputs,
            std::vector<const onnxruntime::NodeArg*>&& subgraphOutputs,
            std::vector<std::shared_ptr<onnxruntime::NodeArg>>&& intermediateNodeArgs,
            std::unordered_map<std::string, GraphNodeProperties>&& partitionNodePropsMap,
            std::vector<ONNX_NAMESPACE::TensorProto>&& ownedInitializers)
        : OpKernel(kernelInfo),
          m_indexedSubGraph(std::move(indexedSubGraph)),
          m_modelPath(modelPath),
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
            // Allocate a persistent resource and initialize the operator
            UINT64 persistentResourceSize = m_compiledExecutionPlanOperator->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_provider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Disabled,
                    m_persistentResource.ReleaseAndGetAddressOf(),
                    m_persistentResourceAllocatorUnk.ReleaseAndGetAddressOf()));

                m_persistentResourceBinding = DML_BUFFER_BINDING { m_persistentResource.Get(), 0, persistentResourceSize };
            }

            ORT_THROW_IF_FAILED(m_provider->InitializeOperator(
                m_compiledExecutionPlanOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(initInputBindings)));

            std::for_each(
                initializeResourceRefs.begin(),
                initializeResourceRefs.end(),
                [&](ComPtr<ID3D12Resource>& resource){ m_winmlProvider->QueueReference(WRAP_GRAPHICS_UNKNOWN(resource).Get()); }
            );
        }

        onnxruntime::Status Compute(onnxruntime::OpKernelContext* kernelContext) const override
        {
            ORT_THROW_HR_IF(E_UNEXPECTED, static_cast<ptrdiff_t>(m_subgraphInputs.size()) != kernelContext->InputCount());

            bool recompileNeeded = m_compiledExecutionPlanOperator == nullptr;

            for (int inputIndex = 0; inputIndex < kernelContext->InputCount(); ++inputIndex)
            {
                const auto& input = kernelContext->RequiredInput<onnxruntime::Tensor>(inputIndex);
                const std::string& inputName = m_subgraphInputs[inputIndex]->Name();
                auto shapeIter = m_inferredInputShapes.find(inputName);

                if (shapeIter == m_inferredInputShapes.end())
                {
                    m_inferredInputShapes[inputName] = input.Shape();
                    recompileNeeded = true;
                }
                else if (shapeIter->second != input.Shape())
                {
                    shapeIter->second = input.Shape();
                    recompileNeeded = true;
                }

                // If we have CPU inputs that are not initializers (i.e. they were computed at runtime), add them to the initializer list
                if (input.Location().device.Type() == OrtDevice::CPU)
                {
                    auto inputProto = onnxruntime::utils::TensorToTensorProto(input, inputName);

                    // We can only avoid recompiling the graph when all CPU inputs are identical
                    auto initializerIter = m_isInitializerTransferable.find(inputName);

                    if (initializerIter != m_isInitializerTransferable.end())
                    {
                        if (initializerIter->second.first->raw_data().length() == inputProto.raw_data().length())
                        {
                            for (int i = 0; i < inputProto.raw_data().length(); ++i)
                            {
                                if (initializerIter->second.first->raw_data()[i] != inputProto.raw_data()[i])
                                {
                                    recompileNeeded = true;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            recompileNeeded = true;
                        }
                    }
                    else
                    {
                        recompileNeeded = true;
                    }

                    m_ownedCpuInputs.push_back(std::make_unique<ONNX_NAMESPACE::TensorProto>(std::move(inputProto)));
                    m_isInitializerTransferable[inputName] = std::make_pair(m_ownedCpuInputs.back().get(), false);
                }
            }

            if (recompileNeeded)
            {
                // Go through all the node args and replace their shapes with the real ones
                for (auto& nodeArg : m_intermediateNodeArgs)
                {
                    auto iter = m_inferredInputShapes.find(nodeArg->Name());
                    if (iter != m_inferredInputShapes.end())
                    {
                        auto tensorShape = *nodeArg->Shape();
                        ORT_THROW_HR_IF(E_UNEXPECTED, tensorShape.dim_size() != static_cast<ptrdiff_t>(iter->second.NumDimensions()));

                        for (int i = 0; i < tensorShape.dim_size(); ++i)
                        {
                            tensorShape.mutable_dim(i)->set_dim_value(iter->second.GetDims()[i]);
                        }

                        nodeArg->SetShape(tensorShape);
                    }
                }

                // Populate input bindings for operator initialization
                const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(m_indexedSubGraph->GetMetaDef()->inputs.size());
                std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> initializeResourceRefs; // For lifetime control
                std::vector<DML_BUFFER_BINDING> initInputBindings(fusedNodeInputCount);
                std::vector<uint8_t> isInputsUploadedByDmlEP(fusedNodeInputCount);
                auto providerImpl = static_cast<const ExecutionProvider*>(Info().GetExecutionProvider())->GetImpl();

                // Convert partitionONNXGraph into DML EP GraphDesc
                ComPtr<IDMLDevice> device;
                ORT_THROW_IF_FAILED(providerImpl->GetDmlDevice(device.GetAddressOf()));
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
                    m_isInitializerTransferable,
                    m_partitionNodePropsMap,
                    providerImpl,
                    m_modelPath,
                    m_subgraphNodePointers,
                    m_subgraphInputs,
                    m_subgraphOutputs,
                    serializedGraphInputIndexToSubgraphInputIndex,
                    serializedGraphLargeConstantNameToSubgraphInputIndex,
                    smallConstantData);

                m_outputShapes = graphDesc.outputShapes;

                // Walk through each graph edge and mark used inputs
                m_inputsUsed.resize(fusedNodeInputCount, false);
                for (auto it = serializedGraphInputIndexToSubgraphInputIndex.begin(); it != serializedGraphInputIndexToSubgraphInputIndex.end(); it++) {
                    m_inputsUsed[it->second] = true;
                }
                for (auto it = serializedGraphLargeConstantNameToSubgraphInputIndex.begin(); it != serializedGraphLargeConstantNameToSubgraphInputIndex.end(); it++) {
                    m_inputsUsed[it->second] = true;
                }

                // Compile the operator
                m_compiledExecutionPlanOperator = DmlGraphFusionHelper::TryCreateCompiledOperator(
                    graphDesc,
                    *m_indexedSubGraph,
                    providerImpl,
                    &serializedGraphInputIndexToSubgraphInputIndex,
                    &serializedGraphLargeConstantNameToSubgraphInputIndex);

                // Queue references to objects which must be kept alive until resulting GPU work completes
                m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());

                TranslateAndCompileGraph(
                    Info(),
                    initializeResourceRefs,
                    initInputBindings);
            }

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

            auto outputTensors = contextWrapper.GetOutputTensors(m_outputShapes);
            ExecuteOperator(
                m_compiledExecutionPlanOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                inputPtrs,
                outputTensors);

            ORT_THROW_IF_FAILED(m_provider->AddUAVBarrier());

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

        mutable std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
        std::shared_ptr<const onnxruntime::IndexedSubGraph> m_indexedSubGraph;
        const onnxruntime::Path& m_modelPath;

        std::vector<std::shared_ptr<onnxruntime::Node>> m_subgraphNodes;
        std::vector<const onnxruntime::NodeArg*> m_subgraphInputs;
        std::vector<const onnxruntime::NodeArg*> m_subgraphOutputs;
        mutable std::vector<std::shared_ptr<onnxruntime::NodeArg>> m_intermediateNodeArgs;
        std::unordered_map<std::string, GraphNodeProperties> m_partitionNodePropsMap;
        std::vector<ONNX_NAMESPACE::TensorProto> m_ownedInitializers;
        mutable std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> m_isInitializerTransferable;
        std::vector<const onnxruntime::Node*> m_subgraphNodePointers;

        // Bindings from previous executions of a re-used command list
        mutable std::vector<std::unique_ptr<ONNX_NAMESPACE::TensorProto>> m_ownedCpuInputs;
        mutable ComPtr<IDMLCompiledOperator> m_compiledExecutionPlanOperator;
        mutable std::vector<bool> m_inputsUsed;
        mutable ComPtr<ID3D12Resource> m_persistentResource;
        mutable ComPtr<IUnknown> m_persistentResourceAllocatorUnk; // Controls when the persistent resource is returned to the allocator
        mutable Windows::AI::MachineLearning::Adapter::EdgeShapes m_outputShapes;
        mutable std::unordered_map<std::string, onnxruntime::TensorShape> m_inferredInputShapes;
    };

    onnxruntime::OpKernel* CreateRuntimeFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        std::shared_ptr<const onnxruntime::IndexedSubGraph> indexedSubGraph,
        const onnxruntime::Path& modelPath,
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
            std::move(subgraphNodes),
            std::move(subgraphInputs),
            std::move(subgraphOutputs),
            std::move(intermediateNodeArgs),
            std::move(partitionNodePropsMap),
            std::move(ownedInitializers)
        );
    }
} // namespace Dml
