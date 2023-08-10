#pragma once

#include "DmlGraphFusionHelper.h"


namespace Dml
{
namespace DmlGraphFusionHelper
{
    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateResource(
        const ExecutionProviderImpl* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;

        D3D12_HEAP_PROPERTIES heapProperties = {
            D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};

        D3D12_RESOURCE_DESC resourceDesc = {D3D12_RESOURCE_DIMENSION_BUFFER,
                                            0,
                                            static_cast<uint64_t>((tensorByteSize + 3) & ~3),
                                            1,
                                            1,
                                            1,
                                            DXGI_FORMAT_UNKNOWN,
                                            {1, 0},
                                            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

        Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
        ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

        ORT_THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(buffer.GetAddressOf())));

        ORT_THROW_IF_FAILED(provider->UploadToResource(buffer.Get(), tensorPtr, tensorByteSize));

        return buffer;
    }

    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateCpuResource(
        const ExecutionProviderImpl* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;

        D3D12_HEAP_PROPERTIES heapProperties = {
            D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0, 0, 0};

        D3D12_RESOURCE_DESC resourceDesc = {D3D12_RESOURCE_DIMENSION_BUFFER,
                                            0,
                                            static_cast<uint64_t>((tensorByteSize + 3) & ~3),
                                            1,
                                            1,
                                            1,
                                            DXGI_FORMAT_UNKNOWN,
                                            {1, 0},
                                            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

        Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
        ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

        ORT_THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(buffer.GetAddressOf())));

        // Map the buffer and copy the data
        void* bufferData = nullptr;
        D3D12_RANGE range = {0, tensorByteSize};
        ORT_THROW_IF_FAILED(buffer->Map(0, &range, &bufferData));
        memcpy(bufferData, tensorPtr, tensorByteSize);
        buffer->Unmap(0, &range);

        return buffer;
    }

    void UnwrapTensor(
        Windows::AI::MachineLearning::Adapter::IWinmlExecutionProvider* winmlProvider,
        const onnxruntime::Tensor* tensor,
        ID3D12Resource** resource,
        uint64_t* allocId)
    {
        IUnknown* allocationUnk = static_cast<IUnknown*>(const_cast<void*>(tensor->DataRaw()));
        Microsoft::WRL::ComPtr<IUnknown> resourceUnk;
        winmlProvider->GetABIDataInterface(false, allocationUnk, &resourceUnk);

        *allocId = winmlProvider->TryGetPooledAllocationId(allocationUnk, 0);

        ORT_THROW_IF_FAILED(resourceUnk->QueryInterface(resource));
    }

    void ProcessInputData(
        const ExecutionProviderImpl* providerImpl,
        const std::vector<uint8_t>& isInputsUploadedByDmlEP,
        const std::vector<DML_INPUT_GRAPH_EDGE_DESC>& inputEdges,
        const gsl::span<const std::string> subGraphInputArgNames,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& initializerNameToInitializerMap,
        onnxruntime::Graph& graph,
        _Out_ std::vector<bool>& inputsUsed,
        _Inout_ std::vector<DML_BUFFER_BINDING>& initInputBindings,
        _Inout_ std::vector<ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
        _Inout_ std::vector<ComPtr<ID3D12Resource>>& initializeResourceRefs,
        _Inout_opt_ std::vector<std::vector<std::byte>>* inputRawData)
    {

        const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(subGraphInputArgNames.size());

        // Determine the last input which uses an initializer, so initializers can be freed incrementally
        // while processing each input in order.
        std::map<const onnx::TensorProto*, uint32_t> initializerToLastInputIndexMap;
        for (uint32_t i = 0; i < fusedNodeInputCount; i++)
        {
            auto iter = initializerNameToInitializerMap.find(subGraphInputArgNames[i]);
            if (iter != initializerNameToInitializerMap.end()) {
                initializerToLastInputIndexMap[iter->second.first] = i;
            }
        }

        // Walk through each graph edge and mark used inputs
        inputsUsed.assign(fusedNodeInputCount, false);
        for (const DML_INPUT_GRAPH_EDGE_DESC& edge : inputEdges) {
            inputsUsed[edge.GraphInputIndex] = true;
        }

        for (uint32_t i = 0; i < initInputBindings.size(); i++)
        {
            bool isInitializerAlreadyRemoved = false;
            // If the input isn't actually used by the graph, nothing ever needs to be bound (either for
            // initialization or execution). So just throw away the transferred initializer and skip this input.
            if (!inputsUsed[i])
            {
                auto iter = initializerNameToInitializerMap.find(subGraphInputArgNames[i]);
                if(iter != initializerNameToInitializerMap.end() && iter->second.second)
                {
                    graph.RemoveInitializedTensor(subGraphInputArgNames[i]);
                }

                if (inputRawData)
                {
                    inputRawData->push_back(std::vector<std::byte>());
                }

                continue;
            }

            // Look for the initializer among those transferred from the graph during partitioning
            auto iter = initializerNameToInitializerMap.find(subGraphInputArgNames[i]);
            if (iter != initializerNameToInitializerMap.end())
            {
                std::byte* tensorPtr = nullptr;
                size_t tensorByteSize = 0;
                std::vector<uint8_t> unpackedExternalTensor;

                std::unique_ptr<std::byte[]> unpackedTensor;

                //auto& initializer = iter->second;
                auto* initializer = iter->second.first;

                // The tensor may be stored as raw data or in typed fields.
                if (initializer->data_location() == onnx::TensorProto_DataLocation_EXTERNAL)
                {
                    THROW_IF_NOT_OK(onnxruntime::utils::UnpackInitializerData(*initializer, graph.ModelPath(), unpackedExternalTensor));
                    tensorPtr = reinterpret_cast<std::byte*>(unpackedExternalTensor.data());
                    tensorByteSize = unpackedExternalTensor.size();
                }
                else if (initializer->has_raw_data())
                {
                    tensorPtr = (std::byte*)(initializer->raw_data().c_str());
                    tensorByteSize = initializer->raw_data().size();
                }
                else
                {
                    std::tie(unpackedTensor, tensorByteSize) = Windows::AI::MachineLearning::Adapter::UnpackTensor(*initializer, graph.ModelPath());
                    tensorPtr = unpackedTensor.get();

                    // Free the initializer if this is the last usage of it.
                    if (initializerToLastInputIndexMap[initializer] == i)
                    {
                        if (iter->second.second)
                        {
                            graph.RemoveInitializedTensor(subGraphInputArgNames[i]);
                            isInitializerAlreadyRemoved = true;
                        }
                    }
                }

                // Tensor sizes in DML must be a multiple of 4 bytes large.
                tensorByteSize = AlignToPow2<size_t>(tensorByteSize, 4);

                if (inputRawData)
                {
                    inputRawData->push_back(std::vector<std::byte>(tensorPtr, tensorPtr + tensorByteSize));
                }

                if (!isInputsUploadedByDmlEP[i])
                {
                    // Store the resource to use during execution
                    ComPtr<ID3D12Resource> defaultBuffer = CreateResource(providerImpl, tensorPtr, tensorByteSize);
                    nonOwnedGraphInputsFromInitializers[i] = defaultBuffer;
                    initializeResourceRefs.push_back(std::move(defaultBuffer));
                }
                else
                {
                    ComPtr<ID3D12Resource> initializeInputBuffer;

                    // D3D_FEATURE_LEVEL_1_0_CORE doesn't support Custom heaps
                    if (providerImpl->IsMcdmDevice())
                    {
                        initializeInputBuffer = CreateResource(providerImpl, tensorPtr, tensorByteSize);
                    }
                    else
                    {
                        initializeInputBuffer = CreateCpuResource(providerImpl, tensorPtr, tensorByteSize);
                    }

                    // Set the binding for operator initialization to the buffer
                    initInputBindings[i].Buffer = initializeInputBuffer.Get();
                    initInputBindings[i].SizeInBytes = tensorByteSize;
                    initializeResourceRefs.push_back(std::move(initializeInputBuffer));
                }

                // Free the initializer if this is the last usage of it.
                if (!isInitializerAlreadyRemoved && initializerToLastInputIndexMap[initializer] == i)
                {
                    if (iter->second.second)
                    {
                        graph.RemoveInitializedTensor(subGraphInputArgNames[i]);
                    }
                }
            }
            else if (inputRawData)
            {
                inputRawData->push_back(std::vector<std::byte>());
            }
        }
    }

    std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>>
    GetInitializerToPartitionMap(
        const onnxruntime::GraphViewer& graph,
        gsl::span<std::unique_ptr<GraphPartition>> partitions
    )
    {
        std::unordered_map<const onnx::TensorProto*, std::vector<uint32_t>> initializerPartitionMap;
        for (uint32_t partitionIndex = 0; partitionIndex < gsl::narrow_cast<uint32_t>(partitions.size()); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            // Skip partitions which have been merged into other partitions
            if (partition->GetRootMergedPartition() != partition.get())
            {
                continue;
            }

            for (const std::string& input : partition->GetInputs())
            {
                const onnx::TensorProto* tensor = nullptr;
                if (graph.GetInitializedTensor(input, tensor))
                {
                    initializerPartitionMap[tensor].push_back(partitionIndex);
                }
            }
        }

        return initializerPartitionMap;
    }

    void ConvertGraphDesc(
        const Dml::GraphDescBuilder::GraphDesc& graphDesc,
        _Out_ DML_GRAPH_DESC& dmlGraphDesc,
        const uint32_t inputCount,
        const uint32_t outputCount,
        _Inout_ std::vector<DML_OPERATOR_GRAPH_NODE_DESC>& dmlOperatorGraphNodes,
        _Inout_ std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges)
    {
        for (size_t i = 0; i < graphDesc.nodes.size(); ++i)
        {
            auto& nodeInfo = graphDesc.nodes[i];
            dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{nodeInfo.op.Get(), nodeInfo.name.data()};
            dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]};
        }

        for (size_t i = 0; i < graphDesc.inputEdges.size(); ++i)
        {
            dmlInputEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &graphDesc.inputEdges[i]};
        }

        for (size_t i = 0; i < graphDesc.outputEdges.size(); ++i)
        {
            dmlOutputEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &graphDesc.outputEdges[i]};
        }

        for (size_t i = 0; i < graphDesc.intermediateEdges.size(); ++i)
        {
            dmlIntermediateEdges[i] =
                DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &graphDesc.intermediateEdges[i]};
        }

        dmlGraphDesc.InputCount = inputCount;
        dmlGraphDesc.OutputCount = outputCount;
        dmlGraphDesc.NodeCount = gsl::narrow_cast<uint32_t>(dmlGraphNodes.size());
        dmlGraphDesc.Nodes = dmlGraphNodes.data();
        dmlGraphDesc.InputEdgeCount = gsl::narrow_cast<uint32_t>(dmlInputEdges.size());
        dmlGraphDesc.InputEdges = dmlInputEdges.data();
        dmlGraphDesc.OutputEdgeCount = gsl::narrow_cast<uint32_t>(dmlOutputEdges.size());
        dmlGraphDesc.OutputEdges = dmlOutputEdges.data();
        dmlGraphDesc.IntermediateEdgeCount = gsl::narrow_cast<uint32_t>(dmlIntermediateEdges.size());
        dmlGraphDesc.IntermediateEdges = dmlIntermediateEdges.data();
    }

    onnxruntime::IndexedSubGraph CreateIndexedSubGraph(
        GraphPartition* partition,
        uint32_t partitionIndex,
        const std::string& partitionKernelPrefix)
    {
        assert(partition->IsDmlGraphPartition());

        onnxruntime::IndexedSubGraph indexedSubGraph;
        // Create a definition for the node.  The name must be unique.
        auto def = std::make_unique<onnxruntime::IndexedSubGraph::MetaDef>();
        def->name = DmlGraphFusionTransformer::DML_GRAPH_FUSION_NODE_NAME_PREFIX + partitionKernelPrefix + std::to_string(partitionIndex);
        def->domain = DmlGraphFusionTransformer::DML_GRAPH_FUSION_NODE_DOMAIN;
        def->since_version = 1;
        def->inputs.insert(def->inputs.begin(), partition->GetInputs().begin(), partition->GetInputs().end());
        def->outputs.insert(def->outputs.begin(), partition->GetOutputs().begin(), partition->GetOutputs().end());

        indexedSubGraph.SetMetaDef(std::move(def));
        indexedSubGraph.nodes = std::move(partition->GetNodeIndices());

        return indexedSubGraph;
    }

    std::unordered_map<std::string, GraphNodeProperties> CreatePartitionNodePropsMap(
        const onnxruntime::Graph& graph,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties>&& graphNodePropertyMap)
    {
        // Populate properties which will be passed to OpKernel for this graph via the function below
        std::unordered_map<std::string, GraphNodeProperties> partitionNodePropsMap;
        for (auto nodeIndex : indexedSubGraph.nodes)
        {
            const onnxruntime::Node* node = graph.GetNode(nodeIndex);

#ifdef PRINT_PARTITON_INFO
            printf("Partition %u\t%s\n", partitionIndex, GraphDescBuilder::GetUniqueNodeName(*node).c_str());
#endif
            partitionNodePropsMap.insert(std::make_pair(
                GraphDescBuilder::GetUniqueNodeName(*node), std::move(graphNodePropertyMap[node])));
        }

#ifdef PRINT_PARTITON_INFO
        printf("\n");
#endif

        return partitionNodePropsMap;
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> TryCreateCompiledOperator(
        const GraphDescBuilder::GraphDesc& graphDesc,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        const ExecutionProviderImpl* providerImpl)
    {
        const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(indexedSubGraph.GetMetaDef()->inputs.size());
        const uint32_t fusedNodeOutputCount = gsl::narrow_cast<uint32_t>(indexedSubGraph.GetMetaDef()->outputs.size());

        // convert DML EP GraphDesc into DML_GRAPH_DESC and create IDMLCompiledOperator
        DML_GRAPH_DESC dmlGraphDesc = {};
        std::vector<DML_OPERATOR_GRAPH_NODE_DESC> dmlOperatorGraphNodes(graphDesc.nodes.size());
        std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes(graphDesc.nodes.size());
        std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges(graphDesc.inputEdges.size());
        std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges(graphDesc.outputEdges.size());
        std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges(graphDesc.intermediateEdges.size());
        ConvertGraphDesc(
            graphDesc,
            dmlGraphDesc,
            fusedNodeInputCount,
            fusedNodeOutputCount,
            dmlOperatorGraphNodes,
            dmlGraphNodes,
            dmlInputEdges,
            dmlOutputEdges,
            dmlIntermediateEdges);

        DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_NONE;
        if (graphDesc.reuseCommandList)
        {
            executionFlags |= DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        }

        // Query DML execution provider to see if metacommands is enabled
        if (!providerImpl->MetacommandsEnabled())
        {
            executionFlags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
        }

        ComPtr<IDMLDevice> device;
        ORT_THROW_IF_FAILED(providerImpl->GetDmlDevice(device.GetAddressOf()));

        ComPtr<IDMLDevice1> device1;
        ORT_THROW_IF_FAILED(device.As(&device1));

        ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator;
        ORT_THROW_IF_FAILED(device1->CompileGraph(
            &dmlGraphDesc,
            executionFlags,
            IID_PPV_ARGS(&compiledExecutionPlanOperator)));

        // UINT32_MAX is currently the maximum number of bytes allowed by D3D12 for the offset of a view over a resource
        if (compiledExecutionPlanOperator->GetBindingProperties().PersistentResourceSize > UINT32_MAX)
        {
            return nullptr;
        }

        return compiledExecutionPlanOperator;
    }

    void FusePartitionAndRegisterKernel(
        onnxruntime::Graph& graph,
        onnxruntime::KernelRegistry* registryForPartitionKernels,
        const std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>>& initializerNameToInitializerMap,
        const ExecutionProviderImpl* providerImpl,
        const onnxruntime::IndexedSubGraph& indexedSubGraph,
        std::vector<uint8_t>&& isInputsUploadedByDmlEP,
        const GraphDescBuilder::GraphDesc& graphDesc,
        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator)
    {
        auto& fusedNode = graph.BeginFuseSubGraph(indexedSubGraph, indexedSubGraph.GetMetaDef()->name);
        fusedNode.SetExecutionProviderType(onnxruntime::kDmlExecutionProvider);

        const uint32_t fusedNodeInputCount = gsl::narrow_cast<uint32_t>(indexedSubGraph.GetMetaDef()->inputs.size());

        // Populate input bindings for operator initialization
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> initializeResourceRefs; // For lifetime control
        std::vector<DML_BUFFER_BINDING> initInputBindings(fusedNodeInputCount);
        std::vector<ComPtr<ID3D12Resource>> nonOwnedGraphInputsFromInitializers(fusedNodeInputCount);

        std::vector<bool> inputsUsed;
        ProcessInputData(
            providerImpl,
            isInputsUploadedByDmlEP,
            graphDesc.inputEdges,
            indexedSubGraph.GetMetaDef()->inputs,
            initializerNameToInitializerMap,
            graph,
            inputsUsed,
            initInputBindings,
            nonOwnedGraphInputsFromInitializers,
            initializeResourceRefs,
            nullptr);

        // lamda captures for the kernel registration
        Windows::AI::MachineLearning::Adapter::EdgeShapes outputShapes;
        ORT_THROW_HR_IF(E_UNEXPECTED, !TryGetStaticOutputShapes(fusedNode, outputShapes));
        bool resuableCommandList = graphDesc.reuseCommandList;
        auto fused_kernel_func = [compiledExecutionPlanOperator,
                                  outputShapes,
                                  resuableCommandList,
                                  nonOwnedGraphInputsFromInitializers,
                                  initializeResourceRefs,
                                  initInputBindings,
                                  isInputsUploadedByDmlEP = std::move(isInputsUploadedByDmlEP),
                                  inputsUsed = std::move(inputsUsed)]
                    (onnxruntime::FuncManager& func_mgr, const onnxruntime::OpKernelInfo& info, std::unique_ptr<onnxruntime::OpKernel>& out) mutable ->onnxruntime::Status
        {
            out.reset(CreateFusedGraphKernel(info,
                                             compiledExecutionPlanOperator,
                                             outputShapes,
                                             resuableCommandList,
                                             nonOwnedGraphInputsFromInitializers,
                                             initializeResourceRefs,
                                             initInputBindings,
                                             std::move(isInputsUploadedByDmlEP),
                                             std::move(inputsUsed)));
            return Status::OK();
        };

        // build the kernel definition on the fly, and register it to the fused_kernel_regisitry.
        onnxruntime::KernelDefBuilder builder;
        builder.SetName(indexedSubGraph.GetMetaDef()->name)
            .SetDomain(indexedSubGraph.GetMetaDef()->domain)
            .SinceVersion(indexedSubGraph.GetMetaDef()->since_version)
            .Provider(onnxruntime::kDmlExecutionProvider);
        ORT_THROW_IF_ERROR(registryForPartitionKernels->Register(builder, fused_kernel_func));

        graph.FinalizeFuseSubGraph(indexedSubGraph, fusedNode);
    }
}
}
