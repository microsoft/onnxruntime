// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlRuntimeFusedGraphKernel.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlGraphFusionHelper.h"
#include "core/providers/dml/DmlExecutionProvider/src/DmlReusedCommandListState.h"

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

        void TranslateAndCompileGraph(const onnxruntime::OpKernelInfo& kernelInfo, std::vector<DML_BUFFER_BINDING> initInputBindings) const
        {
            // Allocate a persistent resource and initialize the operator
            UINT64 persistentResourceSize = m_compiledExecutionPlanOperator->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_provider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Disabled,
                    m_persistentResource.ReleaseAndGetAddressOf(),
                    m_persistentResourceAllocatorUnknown.ReleaseAndGetAddressOf()));

                m_persistentResourceBinding = DML_BUFFER_BINDING { m_persistentResource.Get(), 0, persistentResourceSize };
            }

            ORT_THROW_IF_FAILED(m_provider->InitializeOperator(
                m_compiledExecutionPlanOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(initInputBindings)));
        }

        onnxruntime::Status Compute(onnxruntime::OpKernelContext* kernelContext) const override
        {
            // Release the references from the previous execution since Flush() isn't called for reusable command lists
            auto providerImpl = static_cast<ExecutionProviderImpl*>(m_provider.Get());
            providerImpl->ReleaseCompletedReferences();

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

            const bool graphCaptureEnabled = providerImpl->GraphCaptureEnabled() && providerImpl->GetCurrentGraphAnnotationId() != -1;

            if (graphCaptureEnabled)
            {
                // Only capture the graph the first time
                if (!providerImpl->GraphCaptured(providerImpl->GetCurrentGraphAnnotationId()))
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
                    std::unordered_map<std::string, std::vector<uint32_t>> inferredOutputShapes;

                    const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(m_indexedSubGraph->GetMetaDef()->inputs.size());
                    std::vector<DML_BUFFER_BINDING> initInputBindings(fusedNodeInputCount);
                    std::vector<uint8_t> isInputsUploadedByDmlEP(fusedNodeInputCount);
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
                        smallConstantData,
                        inferredOutputShapes);

                    m_outputShapes = graphDesc.outputShapes;

                    // Walk through each graph edge and mark used inputs
                    m_inputsUsed = std::vector<bool>(fusedNodeInputCount);
                    for (auto it = serializedGraphInputIndexToSubgraphInputIndex.begin(); it != serializedGraphInputIndexToSubgraphInputIndex.end(); it++) {
                        m_inputsUsed[it->second] = true;
                    }
                    for (auto it = serializedGraphLargeConstantNameToSubgraphInputIndex.begin(); it != serializedGraphLargeConstantNameToSubgraphInputIndex.end(); it++) {
                        m_inputsUsed[it->second] = true;
                    }

                    m_isInputsUploadedByDmlEP.resize(fusedNodeInputCount, 0);
                    m_nonOwnedGraphInputsFromInitializers.resize(fusedNodeInputCount);
                    graphDesc.reuseCommandList = true;

                    // Compile the operator
                    m_compiledExecutionPlanOperator = DmlGraphFusionHelper::TryCreateCompiledOperator(
                        graphDesc,
                        *m_indexedSubGraph,
                        providerImpl,
                        &serializedGraphInputIndexToSubgraphInputIndex,
                        &serializedGraphLargeConstantNameToSubgraphInputIndex);

                    // Queue references to objects which must be kept alive until resulting GPU work completes
                    m_winmlProvider->QueueReference(m_compiledExecutionPlanOperator.Get());
                    TranslateAndCompileGraph(Info(), initInputBindings);

                    auto reusableCommandList = DmlGraphFusionHelper::BuildReusableCommandList(
                        m_provider.Get(),
                        m_compiledExecutionPlanOperator.Get(),
                        m_persistentResource.Get(),
                        m_persistentResourceBinding);

                    reusableCommandList->persistentResource = m_persistentResource;
                    reusableCommandList->persistentResourceAllocatorUnknown = m_persistentResourceAllocatorUnknown;

                    // Keep the temporary resource alive since we won't call ExecuteReusableCommandList again, but will merely replay
                    // the graph in the future. Therefore, all executions of the graph will use the same temporary resource that was
                    // allocated here the first time.
                    constexpr bool keepTemporaryResourceAlive = true;

                    DmlGraphFusionHelper::ExecuteReusableCommandList(
                        kernelContext,
                        *reusableCommandList,
                        m_compiledExecutionPlanOperator.Get(),
                        Info(),
                        m_isInputsUploadedByDmlEP,
                        m_inputsUsed,
                        m_nonOwnedGraphInputsFromInitializers,
                        m_outputShapes,
                        m_winmlProvider.Get(),
                        m_provider.Get(),
                        m_persistentResourceAllocatorUnknown.Get(),
                        keepTemporaryResourceAlive);

                    providerImpl->AppendCapturedGraph(providerImpl->GetCurrentGraphAnnotationId(), std::move(reusableCommandList));
                }
            }
            else
            {
                struct IntermediateTensorInfo
                {
                    uint64_t offset;
                    uint64_t size;
                    uint32_t resourceIndex;
                    uint32_t dmlInputIndex;
                };

                std::vector<ComPtr<IDMLCompiledOperator>> compiledOperators;
                std::vector<uint64_t> persistentResourceOffsets;
                uint64_t persistentResourceSize = 0;
                uint64_t temporaryResourceSize = 0;

                uint32_t currentIntermediateResourceIndex = 0;
                std::vector<uint64_t> intermediateResourceSizes = {0};

                std::unordered_map<std::string, uint32_t> inputNameToGlobalIndex;
                for (uint32_t i = 0; i < m_subgraphInputs.size(); ++i)
                {
                    inputNameToGlobalIndex.emplace(m_subgraphInputs[i]->Name(), i);
                }

                std::unordered_map<std::string, uint32_t> outputNameToGlobalIndex;
                for (uint32_t i = 0; i < m_subgraphOutputs.size(); ++i)
                {
                    outputNameToGlobalIndex.emplace(m_subgraphOutputs[i]->Name(), i);
                }

                std::unordered_map<std::string, IntermediateTensorInfo> nodeArgNameToResourceOffset;

                std::vector<std::vector<DML_BUFFER_BINDING>> inputBindings(m_subgraphNodePointers.size());
                std::vector<std::vector<DML_BINDING_DESC>> inputBindingDescs(m_subgraphNodePointers.size());
                std::vector<std::vector<DML_BUFFER_BINDING>> outputBindings(m_subgraphNodePointers.size());
                std::vector<std::vector<DML_BINDING_DESC>> outputBindingDescs(m_subgraphNodePointers.size());
                std::unordered_map<std::string, std::vector<uint32_t>> inferredOutputShapes;

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

                for (uint32_t nodeIndex = 0; nodeIndex < m_subgraphNodePointers.size(); ++nodeIndex)
                {
                    auto node = m_subgraphNodePointers[nodeIndex];

                    const auto& inputArgs = node->InputDefs();
                    std::vector<const onnxruntime::NodeArg*> subgraphInputs(inputArgs.begin(), inputArgs.end());

                    const auto& outputArgs = node->OutputDefs();
                    std::vector<const onnxruntime::NodeArg*> subgraphOutputs(outputArgs.begin(), outputArgs.end());

                    std::vector<uint8_t> isInputsUploadedByDmlEP(node->InputDefs().size());
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
                        gsl::make_span(&node, 1),
                        subgraphInputs,
                        subgraphOutputs,
                        serializedGraphInputIndexToSubgraphInputIndex,
                        serializedGraphLargeConstantNameToSubgraphInputIndex,
                        smallConstantData,
                        inferredOutputShapes);

                    auto inputsUsed = std::vector<bool>(node->InputDefs().size());
                    for (auto it = serializedGraphInputIndexToSubgraphInputIndex.begin(); it != serializedGraphInputIndexToSubgraphInputIndex.end(); it++) {
                        inputsUsed[it->second] = true;
                    }
                    for (auto it = serializedGraphLargeConstantNameToSubgraphInputIndex.begin(); it != serializedGraphLargeConstantNameToSubgraphInputIndex.end(); it++) {
                        inputsUsed[it->second] = true;
                    }

                    // Compile the operator
                    auto compiledOp = DmlGraphFusionHelper::TryCreateCompiledOperator(
                        graphDesc,
                        *m_indexedSubGraph,
                        providerImpl,
                        &serializedGraphInputIndexToSubgraphInputIndex,
                        &serializedGraphLargeConstantNameToSubgraphInputIndex);

                    if (compiledOp->GetBindingProperties().PersistentResourceSize > 0)
                    {
                        persistentResourceOffsets.push_back(persistentResourceSize);
                        persistentResourceSize += compiledOp->GetBindingProperties().PersistentResourceSize;
                    }
                    else
                    {
                        persistentResourceOffsets.push_back(0);
                    }

                    temporaryResourceSize = std::max(temporaryResourceSize, compiledOp->GetBindingProperties().TemporaryResourceSize);
                    compiledOperators.push_back(std::move(compiledOp));

                    inputBindings[nodeIndex].resize(graphDesc.InputCount);
                    inputBindingDescs[nodeIndex].resize(graphDesc.InputCount);

                    for (uint32_t onnxInputIndex = 0; onnxInputIndex < inputArgs.size(); ++onnxInputIndex)
                    {
                        auto usedInputIter = serializedGraphInputIndexToSubgraphInputIndex.find(onnxInputIndex);

                        if (usedInputIter != serializedGraphInputIndexToSubgraphInputIndex.end())
                        {
                            const uint32_t dmlInputIndex = usedInputIter->second;
                            auto globalInputIter = inputNameToGlobalIndex.find(inputArgs[onnxInputIndex]->Name());

                            if (globalInputIter != inputNameToGlobalIndex.end())
                            {
                                assert(kernelContext->InputType(gsl::narrow_cast<int>(globalInputIter->second))->IsTensorType());
                                const onnxruntime::Tensor* tensor = kernelContext->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(globalInputIter->second));

                                uint64_t allocId;
                                DmlGraphFusionHelper::UnwrapTensor(m_winmlProvider.Get(), tensor, &inputBindings[nodeIndex][dmlInputIndex].Buffer, &allocId);
                                inputBindings[nodeIndex][dmlInputIndex].Buffer->Release(); // Avoid holding an additional reference
                                inputBindings[nodeIndex][dmlInputIndex].SizeInBytes = DmlGraphFusionHelper::AlignToPow2<size_t>(tensor->SizeInBytes(), 4);
                                inputBindingDescs[nodeIndex][dmlInputIndex] = {DML_BINDING_TYPE_BUFFER, &inputBindings[dmlInputIndex]};
                            }
                            else
                            {
                                auto globalOutputIter = outputNameToGlobalIndex.find(inputArgs[onnxInputIndex]->Name());

                                // The input may also be a global output, in which case we need to fetch it
                                if (globalOutputIter != outputNameToGlobalIndex.end())
                                {
                                    const auto& outputShape = inferredOutputShapes[inputArgs[onnxInputIndex]->Name()];
                                    std::vector<int64_t> int64OutputShape(outputShape.begin(), outputShape.end());

                                    onnxruntime::Tensor* tensor = kernelContext->Output(
                                        static_cast<int>(globalOutputIter->second),
                                        onnxruntime::TensorShape::FromExistingBuffer(int64OutputShape));

                                    uint64_t allocId;
                                    DmlGraphFusionHelper::UnwrapTensor(m_winmlProvider.Get(), tensor, &inputBindings[nodeIndex][dmlInputIndex].Buffer, &allocId);
                                    inputBindings[nodeIndex][dmlInputIndex].Buffer->Release(); // Avoid holding an additional reference
                                    inputBindings[nodeIndex][dmlInputIndex].SizeInBytes = DmlGraphFusionHelper::AlignToPow2<size_t>(tensor->SizeInBytes(), 4);
                                    inputBindingDescs[nodeIndex][dmlInputIndex] = {DML_BINDING_TYPE_BUFFER, &inputBindings[dmlInputIndex]};
                                }
                                else
                                {
                                    auto iter = nodeArgNameToResourceOffset.find(inputArgs[onnxInputIndex]->Name());

                                    if (iter == nodeArgNameToResourceOffset.end())
                                    {
                                        // If the arg is not found, then we record its offset and size into the merged resource that we'll allocate later
                                        auto dtype = onnx::TensorProto_DataType(inputArgs[onnxInputIndex]->TypeAsProto()->tensor_type().elem_type());
                                        const auto& inputShape = inferredOutputShapes[inputArgs[onnxInputIndex]->Name()];
                                        auto mlDataType = Windows::AI::MachineLearning::Adapter::ToMLTensorDataType(dtype);
                                        auto dmlDataType = GetDmlDataTypeFromMlDataType(mlDataType);
                                        const uint64_t sizeInBytes = DMLCalcBufferTensorSize(dmlDataType, gsl::narrow_cast<uint32_t>(inputShape.size()), inputShape.data(), nullptr);

                                        IntermediateTensorInfo intermediateTensorInfo{};
                                        intermediateTensorInfo.offset = intermediateResourceSizes.back();
                                        intermediateTensorInfo.size = sizeInBytes;
                                        intermediateTensorInfo.resourceIndex = currentIntermediateResourceIndex;
                                        intermediateTensorInfo.dmlInputIndex = dmlInputIndex;

                                        intermediateResourceSizes.back() += sizeInBytes;

                                        // If the offset is bigger than UINT32_MAX, we need to move on to another resource for the following allocations since
                                        // views cannot have an offset in bytes bigger than UINT32_MAX
                                        if (intermediateResourceSizes.back() > UINT32_MAX)
                                        {
                                            intermediateResourceSizes.push_back(0);
                                            ++currentIntermediateResourceIndex;
                                        }

                                        nodeArgNameToResourceOffset.emplace(inputArgs[onnxInputIndex]->Name(), std::move(intermediateTensorInfo));
                                    }
                                }
                            }
                        }
                    }

                    outputBindings[nodeIndex].resize(graphDesc.OutputCount);
                    outputBindingDescs[nodeIndex].resize(graphDesc.OutputCount);

                    for (uint32_t i = 0; i < outputArgs.size(); ++i)
                    {
                        auto globalOutputIter = outputNameToGlobalIndex.find(outputArgs[i]->Name());
                        if (globalOutputIter != outputNameToGlobalIndex.end())
                        {
                            const auto& outputShape = inferredOutputShapes[outputArgs[i]->Name()];
                            std::vector<int64_t> int64OutputShape(outputShape.begin(), outputShape.end());

                            onnxruntime::Tensor* tensor = kernelContext->Output(
                                static_cast<int>(globalOutputIter->second),
                                onnxruntime::TensorShape::FromExistingBuffer(int64OutputShape));

                            uint64_t allocId;
                            DmlGraphFusionHelper::UnwrapTensor(m_winmlProvider.Get(), tensor, &outputBindings[nodeIndex][i].Buffer, &allocId);
                            outputBindings[nodeIndex][i].Buffer->Release(); // Avoid holding an additional reference
                            outputBindings[nodeIndex][i].SizeInBytes = DmlGraphFusionHelper::AlignToPow2<size_t>(tensor->SizeInBytes(), 4);
                            outputBindingDescs[nodeIndex][i] = {DML_BINDING_TYPE_BUFFER, &outputBindings[i]};
                        }
                    }
                }

                ComPtr<ID3D12Device> d3d12_device;
                ORT_THROW_IF_FAILED(providerImpl->GetD3DDevice(d3d12_device.GetAddressOf()));

                // Create the combined intermediate resources
                std::vector<ComPtr<ID3D12Resource>> intermediateResources(intermediateResourceSizes.size());
                for (uint32_t i = 0; i < intermediateResourceSizes.size(); ++i)
                {
                    if (intermediateResourceSizes[i] > 0)
                    {
                        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(intermediateResourceSizes[i], D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                        ORT_THROW_IF_FAILED(d3d12_device->CreateCommittedResource(
                            unmove_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)),
                            D3D12_HEAP_FLAG_NONE,
                            &buffer,
                            D3D12_RESOURCE_STATE_COMMON,
                            nullptr,
                            IID_GRAPHICS_PPV_ARGS(intermediateResources[i].GetAddressOf())
                        ));

                        m_winmlProvider->QueueReference(intermediateResources[i].Get());
                    }
                }

                // Now that the intermediate resources have been created, we can reiterate through the nodes to set the missing input/output bindings
                for (uint32_t nodeIndex = 0; nodeIndex < m_subgraphNodePointers.size(); ++nodeIndex)
                {
                    auto node = m_subgraphNodePointers[nodeIndex];

                    const auto& inputArgs = node->InputDefs();
                    for (auto inputArg : inputArgs)
                    {
                        auto iter = nodeArgNameToResourceOffset.find(inputArg->Name());
                        if (iter != nodeArgNameToResourceOffset.end())
                        {
                            inputBindings[nodeIndex][iter->second.dmlInputIndex].Buffer = intermediateResources[iter->second.resourceIndex].Get();
                            inputBindings[nodeIndex][iter->second.dmlInputIndex].SizeInBytes = iter->second.size;
                            inputBindingDescs[nodeIndex][iter->second.dmlInputIndex] = {DML_BINDING_TYPE_BUFFER, &inputBindings[iter->second.dmlInputIndex]};
                        }
                    }

                    const auto& outputArgs = node->OutputDefs();
                    for (uint32_t i = 0; i < outputArgs.size(); ++i)
                    {
                        auto iter = nodeArgNameToResourceOffset.find(outputArgs[i]->Name());
                        if (iter != nodeArgNameToResourceOffset.end())
                        {
                            outputBindings[nodeIndex][i].Buffer = intermediateResources[iter->second.resourceIndex].Get();
                            outputBindings[nodeIndex][i].SizeInBytes = iter->second.size;
                            outputBindingDescs[nodeIndex][i] = {DML_BINDING_TYPE_BUFFER, &outputBindings[i]};
                        }
                    }
                }

                ComPtr<ID3D12Resource> persistentResource;
                ComPtr<IUnknown> persistentResourceUnknown;

                if (persistentResourceSize > 0)
                {
                    ORT_THROW_IF_FAILED(m_provider->AllocatePooledResource(
                        static_cast<size_t>(persistentResourceSize),
                        AllocatorRoundingMode::Disabled,
                        persistentResource.GetAddressOf(),
                        persistentResourceUnknown.GetAddressOf()));

                    m_winmlProvider->QueueReference(persistentResourceUnknown.Get());
                }

                for (uint32_t i = 0; i < compiledOperators.size(); ++i)
                {
                    DML_BUFFER_BINDING persistentResourceBinding{};

                    if (compiledOperators[i]->GetBindingProperties().PersistentResourceSize > 0)
                    {
                        persistentResourceBinding = DML_BUFFER_BINDING {
                            m_persistentResource.Get(),
                            persistentResourceOffsets[i],
                            compiledOperators[i]->GetBindingProperties().PersistentResourceSize,
                        };
                    }

                    // TODO (pavignol): Provide a temporary resource
                    ORT_THROW_IF_FAILED(m_provider->ExecuteOperator(
                        compiledOperators[i].Get(),
                        &persistentResourceBinding,
                        inputBindingDescs[i],
                        outputBindingDescs[i]));

                    m_winmlProvider->QueueReference(compiledOperators[i].Get());
                }
            }

            return onnxruntime::Status::OK();
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
        mutable ComPtr<IUnknown> m_persistentResourceAllocatorUnknown; // Controls when the persistent resource is returned to the allocator
        mutable Windows::AI::MachineLearning::Adapter::EdgeShapes m_outputShapes;
        mutable std::unordered_map<std::string, onnxruntime::TensorShape> m_inferredInputShapes;
        mutable std::vector<uint8_t> m_isInputsUploadedByDmlEP;
        mutable std::vector<ComPtr<ID3D12Resource>> m_nonOwnedGraphInputsFromInitializers;
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
