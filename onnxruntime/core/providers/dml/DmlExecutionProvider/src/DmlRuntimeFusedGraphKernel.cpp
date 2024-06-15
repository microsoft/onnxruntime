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

        void ResetGraphsInfo() const
        {
            m_graphsInfo.clear();

            GraphInfo graphInfo;

            for (uint32_t i = 0; i < m_subgraphInputs.size(); ++i)
            {
                graphInfo.inputs.emplace_back(GraphInputInfo{m_subgraphInputs[i], std::optional<uint32_t>(i)});
            }

            for (uint32_t i = 0; i < m_subgraphOutputs.size(); ++i)
            {
                graphInfo.outputs.emplace_back(GraphOutputInfo{m_subgraphOutputs[i], std::optional<uint32_t>(i)});
            }

            for (const auto& node : m_subgraphNodes)
            {
                graphInfo.nodes.push_back(node.get());
            }

            m_graphsInfo.push_back(std::move(graphInfo));
        }

        void TranslateAndCompileGraph(GraphInfo& graphInfo, std::vector<DML_BUFFER_BINDING> initInputBindings) const
        {
            // Allocate a persistent resource and initialize the operator
            UINT64 persistentResourceSize = graphInfo.compiledOp->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_provider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Disabled,
                    graphInfo.persistentResource.ReleaseAndGetAddressOf(),
                    graphInfo.persistentResourceAllocatorUnknown.ReleaseAndGetAddressOf()));

                graphInfo.persistentResourceBinding = DML_BUFFER_BINDING { graphInfo.persistentResource.Get(), 0, persistentResourceSize };
            }

            ORT_THROW_IF_FAILED(m_provider->InitializeOperator(
                graphInfo.compiledOp.Get(),
                graphInfo.persistentResourceBinding ? &*graphInfo.persistentResourceBinding : nullptr,
                gsl::make_span(initInputBindings)));
        }

        std::tuple<GraphInfo, GraphInfo> SplitGraph(
            onnxruntime::OpKernelContext* kernelContext,
            const GraphInfo& parentGraph,
            const std::unordered_map<std::string, std::vector<uint32_t>>& inferredOutputShapes) const
        {
            GraphInfo firstGraph{};
            firstGraph.nodes = std::vector<onnxruntime::Node*>(parentGraph.nodes.begin(), parentGraph.nodes.begin() + parentGraph.nodes.size() / 2);

            GraphInfo secondGraph{};
            secondGraph.nodes = std::vector<onnxruntime::Node*>(parentGraph.nodes.begin() + parentGraph.nodes.size() / 2, parentGraph.nodes.end());

            std::unordered_set<std::string> firstGraphInputArgs;
            std::unordered_map<std::string, const onnxruntime::NodeArg*> firstGraphOutputArgs;
            for (auto node : firstGraph.nodes)
            {
                auto inputDefs = node->InputDefs();
                for (auto& inputDef : inputDefs)
                {
                    firstGraphInputArgs.insert(inputDef->Name());
                }

                auto outputDefs = node->OutputDefs();
                for (auto& outputDef : outputDefs)
                {
                    firstGraphOutputArgs.emplace(outputDef->Name(), outputDef);
                }
            }

            std::unordered_set<std::string> secondGraphInputArgs;
            std::unordered_set<std::string> secondGraphOutputArgs;
            for (auto node : secondGraph.nodes)
            {
                auto inputDefs = node->InputDefs();
                for (auto& inputDef : inputDefs)
                {
                    secondGraphInputArgs.insert(inputDef->Name());
                }

                auto outputDefs = node->OutputDefs();
                for (auto& outputDef : outputDefs)
                {
                    secondGraphOutputArgs.insert(outputDef->Name());
                }
            }

            // Set the inputs of the first and second graphs that are also inputs of the parent graph
            for (const auto& parentGraphInput : parentGraph.inputs)
            {
                if (firstGraphInputArgs.count(parentGraphInput.inputArg->Name()))
                {
                    firstGraph.inputs.push_back(parentGraphInput);
                }

                if (secondGraphInputArgs.count(parentGraphInput.inputArg->Name()))
                {
                    secondGraph.inputs.push_back(parentGraphInput);
                }
            }

            // Set the outputs of the first and second graphs that are also outputs of the parent graph
            std::unordered_set<std::string> parentGraphOutputs;
            for (const auto& parentGraphOutput : parentGraph.outputs)
            {
                if (firstGraphOutputArgs.count(parentGraphOutput.outputArg->Name()))
                {
                    firstGraph.outputs.push_back(parentGraphOutput);

                    // If an output of the first graph is both a global output and an input of the second graph,
                    // we can assign it as an input of the second graph without allocating additional memory
                    if (secondGraphInputArgs.count(parentGraphOutput.outputArg->Name()))
                    {
                        GraphInputInfo newSecondGraphInput{};
                        newSecondGraphInput.inputArg = parentGraphOutput.outputArg;
                        newSecondGraphInput.globalOutputIndex = parentGraphOutput.globalOutputIndex;
                        newSecondGraphInput.ownedInputTensor = parentGraphOutput.ownedOutputTensor;
                        secondGraph.inputs.push_back(std::move(newSecondGraphInput));
                    }
                }

                if (secondGraphOutputArgs.count(parentGraphOutput.outputArg->Name()))
                {
                    secondGraph.outputs.push_back(parentGraphOutput);
                }

                parentGraphOutputs.insert(parentGraphOutput.outputArg->Name());
            }

            // Finally, set the outputs of the first graph that are also inputs of the second graph and NOT outputs of the
            // parent graph. We also need to allocate tensors for these intermediate outputs since they are brand new inter-graph
            // tensors that didn't exist before.
            for (auto firstGraphOutputArg : firstGraphOutputArgs)
            {
                if (secondGraphInputArgs.count(firstGraphOutputArg.first) && !parentGraphOutputs.count(firstGraphOutputArg.first))
                {
                    auto mlDataType = onnxruntime::DataTypeImpl::TypeFromProto(*firstGraphOutputArg.second->TypeAsProto());
                    const onnxruntime::TensorTypeBase* tensorTypeBase = mlDataType->AsTensorType();

                    const auto& shape = inferredOutputShapes.at(firstGraphOutputArg.first);
                    std::vector<int64_t> int64Shape(shape.begin(), shape.end());

                    onnxruntime::AllocatorPtr allocator;
                    ORT_THROW_IF_ERROR(kernelContext->GetTempSpaceAllocator(&allocator));

                    GraphOutputInfo newFirstGraphOutput{};
                    newFirstGraphOutput.outputArg = firstGraphOutputArg.second;
                    newFirstGraphOutput.ownedOutputTensor = std::make_shared<onnxruntime::Tensor>(tensorTypeBase->GetElementType(), onnxruntime::TensorShape(int64Shape), allocator);

                    GraphInputInfo newSecondGraphInput{};
                    newSecondGraphInput.inputArg = firstGraphOutputArg.second;
                    newSecondGraphInput.ownedInputTensor = newFirstGraphOutput.ownedOutputTensor;

                    firstGraph.outputs.push_back(std::move(newFirstGraphOutput));
                    secondGraph.inputs.push_back(std::move(newSecondGraphInput));
                }
            }

            return std::make_tuple(std::move(firstGraph), std::move(secondGraph));
        }

        onnxruntime::Status Compute(onnxruntime::OpKernelContext* kernelContext) const override
        {
            // Release the references from the previous execution since Flush() isn't called for reusable command lists
            auto providerImpl = static_cast<ExecutionProviderImpl*>(m_provider.Get());
            providerImpl->ReleaseCompletedReferences();

            ORT_THROW_HR_IF(E_UNEXPECTED, static_cast<ptrdiff_t>(m_subgraphInputs.size()) != kernelContext->InputCount());

            bool recompileNeeded = m_graphsInfo.empty();

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
                ResetGraphsInfo();

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
                std::vector<uint8_t> isInputsUploadedByDmlEP(fusedNodeInputCount);

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
                std::vector<std::unique_ptr<std::byte[]>> smallConstantData;

                uint32_t graphInfoIndex = 0;

                std::unordered_map<std::string, std::vector<uint32_t>> inferredOutputShapes;
                m_outputShapes.Reset(0);

                while (graphInfoIndex < m_graphsInfo.size())
                {
                    std::vector<const onnxruntime::NodeArg*> inputArgs;
                    inputArgs.reserve(m_graphsInfo[graphInfoIndex].inputs.size());

                    std::vector<const onnxruntime::NodeArg*> outputArgs;
                    outputArgs.reserve(m_graphsInfo[graphInfoIndex].outputs.size());

                    for (auto graphInput : m_graphsInfo[graphInfoIndex].inputs)
                    {
                        inputArgs.push_back(graphInput.inputArg);
                    }

                    for (auto graphOutput : m_graphsInfo[graphInfoIndex].outputs)
                    {
                        outputArgs.push_back(graphOutput.outputArg);
                    }

                    std::unordered_map<uint32_t, uint32_t> serializedGraphInputIndexToSubgraphInputIndex;
                    std::unordered_map<std::string_view, uint32_t> serializedGraphLargeConstantNameToSubgraphInputIndex;
                    GraphDescBuilder::GraphDesc graphDesc = GraphDescBuilder::BuildGraphDesc(
                        isInputsUploadedByDmlEP.data(),
                        isInputsUploadedByDmlEP.size(),
                        m_isInitializerTransferable,
                        m_partitionNodePropsMap,
                        providerImpl,
                        m_modelPath,
                        m_graphsInfo[graphInfoIndex].nodes,
                        inputArgs,
                        outputArgs,
                        serializedGraphInputIndexToSubgraphInputIndex,
                        serializedGraphLargeConstantNameToSubgraphInputIndex,
                        smallConstantData,
                        inferredOutputShapes);

                    if (m_outputShapes.EdgeCount() == 0)
                    {
                        m_outputShapes = graphDesc.outputShapes;
                    }

                    // Walk through each graph edge and mark used inputs, but only for the first graph (it will be reused even if the graph is split)
                    m_graphsInfo[graphInfoIndex].inputsUsed.resize(inputArgs.size());
                    for (auto it = serializedGraphInputIndexToSubgraphInputIndex.begin(); it != serializedGraphInputIndexToSubgraphInputIndex.end(); it++)
                    {
                        m_graphsInfo[graphInfoIndex].inputsUsed[it->second] = true;
                    }
                    for (auto it = serializedGraphLargeConstantNameToSubgraphInputIndex.begin(); it != serializedGraphLargeConstantNameToSubgraphInputIndex.end(); it++)
                    {
                        m_graphsInfo[graphInfoIndex].inputsUsed[it->second] = true;
                    }

                    m_isInputsUploadedByDmlEP.resize(fusedNodeInputCount, 0);
                    m_nonOwnedGraphInputsFromInitializers.resize(fusedNodeInputCount);
                    graphDesc.reuseCommandList = true;

                    // Compile the operator
                    m_graphsInfo[graphInfoIndex].compiledOp = DmlGraphFusionHelper::TryCreateCompiledOperator(
                        graphDesc,
                        gsl::narrow_cast<uint32_t>(inputArgs.size()),
                        gsl::narrow_cast<uint32_t>(outputArgs.size()),
                        providerImpl,
                        &serializedGraphInputIndexToSubgraphInputIndex,
                        &serializedGraphLargeConstantNameToSubgraphInputIndex);

                    if (!m_graphsInfo[graphInfoIndex].compiledOp)
                    {
                        // Split the graph in half, replace the current graph with the split graph and insert the second graph right after
                        auto [firstGraph, secondGraph] = SplitGraph(kernelContext, m_graphsInfo[graphInfoIndex], inferredOutputShapes);
                        m_graphsInfo[graphInfoIndex] = std::move(firstGraph);
                        m_graphsInfo.insert(m_graphsInfo.begin() + graphInfoIndex + 1, std::move(secondGraph));
                    }
                    else
                    {
                        ++graphInfoIndex;
                    }
                }

                for (auto& graphInfo : m_graphsInfo)
                {
                    std::vector<DML_BUFFER_BINDING> initInputBindings(graphInfo.inputs.size());
                    TranslateAndCompileGraph(graphInfo, initInputBindings);
                }

                std::vector<DML_BUFFER_BINDING> inputBindings(kernelContext->InputCount());
                std::vector<DML_BINDING_DESC> inputBindingDescs(kernelContext->InputCount());

                m_reusedCommandLists.clear();
            }

            uint64_t temporaryResourceSize = 0;
            for (auto& graphInfo : m_graphsInfo)
            {
                temporaryResourceSize = std::max(temporaryResourceSize, graphInfo.compiledOp->GetBindingProperties().TemporaryResourceSize);
            }

            ComPtr<ID3D12Device> d3d12Device;
            ORT_THROW_IF_FAILED(providerImpl->GetD3DDevice(d3d12Device.GetAddressOf()));

            ComPtr<ID3D12Resource> temporaryResource;
            auto buffer = CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            ORT_THROW_IF_FAILED(d3d12Device->CreateCommittedResource(
                unmove_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)),
                D3D12_HEAP_FLAG_NONE,
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(temporaryResource.GetAddressOf())
            ));

            m_winmlProvider->QueueReference(temporaryResource.Get());

            // When we are capturing a graph, we don't pool the command list and instead transfer it to the execution provider. Captured graph
            // have the same bindings for their entire lifetime.
            if (providerImpl->GraphCaptureEnabled() && providerImpl->GetCurrentGraphAnnotationId() != -1 && !providerImpl->GraphCaptured(providerImpl->GetCurrentGraphAnnotationId()))
            {
                auto reusableCommandList = DmlGraphFusionHelper::BuildReusableCommandList(m_provider.Get(), m_graphsInfo);

                // Keep the temporary resource alive since we won't call ExecuteReusableCommandList again, but will merely replay
                // the graph in the future. Therefore, all executions of the graph will use the same temporary resource that was
                // allocated here the first time.
                constexpr bool keepTemporaryResourceAlive = true;

                DmlGraphFusionHelper::ExecuteReusableCommandList(
                    kernelContext,
                    *reusableCommandList,
                    Info(),
                    m_isInputsUploadedByDmlEP,
                    m_nonOwnedGraphInputsFromInitializers,
                    m_outputShapes,
                    m_winmlProvider.Get(),
                    m_provider.Get(),
                    keepTemporaryResourceAlive,
                    temporaryResource.Get());

                providerImpl->AppendCapturedGraph(providerImpl->GetCurrentGraphAnnotationId(), std::move(reusableCommandList));
            }
            else
            {
                if (m_reusedCommandLists.empty() ||
                    m_reusedCommandLists.front()->fence && m_reusedCommandLists.front()->fence->GetCompletedValue() < m_reusedCommandLists.front()->completionValue)
                {
                    m_reusedCommandLists.push_front(DmlGraphFusionHelper::BuildReusableCommandList(m_provider.Get(), m_graphsInfo));
                }

                // We don't need to keep a reference on the temporary resource once we have recorded into the command list, so the
                // memory can be reused by the allocator
                constexpr bool keepTemporaryResourceAlive = false;

                DmlGraphFusionHelper::ExecuteReusableCommandList(
                    kernelContext,
                    *m_reusedCommandLists.front(),
                    Info(),
                    m_isInputsUploadedByDmlEP,
                    m_nonOwnedGraphInputsFromInitializers,
                    m_outputShapes,
                    m_winmlProvider.Get(),
                    m_provider.Get(),
                    keepTemporaryResourceAlive,
                    temporaryResource.Get());

                m_reusedCommandLists.push_back(std::move(m_reusedCommandLists.front()));
                m_reusedCommandLists.pop_front();
            }

            return onnxruntime::Status::OK();
        }

    private:
        ComPtr<IWinmlExecutionProvider> m_winmlProvider;
        ComPtr<Dml::IExecutionProvider> m_provider;

        std::shared_ptr<const onnxruntime::IndexedSubGraph> m_indexedSubGraph;
        const onnxruntime::Path& m_modelPath;

        std::vector<std::shared_ptr<onnxruntime::Node>> m_subgraphNodes;
        std::vector<const onnxruntime::NodeArg*> m_subgraphInputs;
        std::vector<const onnxruntime::NodeArg*> m_subgraphOutputs;
        mutable std::vector<GraphInfo> m_graphsInfo;
        mutable std::vector<std::shared_ptr<onnxruntime::NodeArg>> m_intermediateNodeArgs;
        std::unordered_map<std::string, GraphNodeProperties> m_partitionNodePropsMap;
        std::vector<ONNX_NAMESPACE::TensorProto> m_ownedInitializers;
        mutable std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> m_isInitializerTransferable;
        std::vector<const onnxruntime::Node*> m_subgraphNodePointers;

        // Bindings from previous executions of a re-used command list
        mutable std::vector<std::unique_ptr<ONNX_NAMESPACE::TensorProto>> m_ownedCpuInputs;
        mutable Windows::AI::MachineLearning::Adapter::EdgeShapes m_outputShapes;
        mutable std::unordered_map<std::string, onnxruntime::TensorShape> m_inferredInputShapes;
        mutable std::deque<std::unique_ptr<DmlReusedCommandListState>> m_reusedCommandLists;
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
