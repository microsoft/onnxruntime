// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlOperator.h"

namespace Dml
{

    /*static*/ const uint32_t DmlOperator::zeroArray[8] = {};

    DmlOperator::DmlOperator(const MLOperatorKernelCreationContext& kernelInfo)
    {
        ML_CHECK_HRESULT(kernelInfo.GetExecutionInterface().As(&m_executionProvider));
        ML_CHECK_HRESULT(m_executionProvider->GetDmlDevice(/*out*/ m_dmlDevice.GetAddressOf()));
    }

    void DmlOperator::SetDmlOperatorDesc(
        const DML_OPERATOR_DESC& operatorDesc,
        const MLOperatorKernelCreationContext& kernelInfo
        )
    {
        // Initialize should only be called once.
        assert(m_compiledOperator == nullptr);

        // DML doesn't support empty tensors. If an operator is still executable with empty tensors, the empty tensors
        // should be removed or massaged depending on the definition.
        for (const TensorDesc& desc : m_inputTensorDescs)
        {
            if (OperatorHelper::ContainsEmptyDimensions(desc.GetSizes()))
            {
                return;
            }
        }

        for (const TensorDesc& desc : m_outputTensorDescs)
        {
            if (OperatorHelper::ContainsEmptyDimensions(desc.GetSizes()))
            {
                return;
            }
        }

        ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));

        if (contextPrivate->IsDmlGraphNode())
        {
            MLOperatorGraphDesc operatorGraphDesc = {};
            operatorGraphDesc.nodeCount = 1;
            const DML_OPERATOR_DESC* opDescs{&operatorDesc};
            operatorGraphDesc.nodesAsOpDesc = &opDescs;

            std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
            for (uint32_t inputIndex = 0; inputIndex < m_kernelInputIndices.size(); inputIndex++)
            {
                if (m_kernelInputIndices[inputIndex].has_value())
                {
                    DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
                    inputEdge.GraphInputIndex = *m_kernelInputIndices[inputIndex];
                    inputEdge.ToNodeIndex = 0;
                    inputEdge.ToNodeInputIndex = inputIndex;
                    inputEdges.push_back(inputEdge);
                }
            }
            operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
            operatorGraphDesc.inputEdges = inputEdges.data();


            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
            for (uint32_t outputIndex = 0; outputIndex < m_kernelOutputIndices.size(); outputIndex++)
            {
                if (m_kernelOutputIndices[outputIndex].has_value())
                {
                    DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
                    outputEdge.FromNodeIndex = 0;
                    outputEdge.FromNodeOutputIndex = outputIndex;
                    outputEdge.GraphOutputIndex = (*m_kernelOutputIndices[outputIndex]);
                    outputEdges.push_back(outputEdge);
                }
            }
            operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
            operatorGraphDesc.outputEdges = outputEdges.data();

            ORT_THROW_IF_FAILED(contextPrivate->SetDmlOperator(&operatorGraphDesc));
        }
        else
        {
            auto operatorDescCopy = operatorDesc;

            // TODO: Change as new header is ingested
            if (operatorDescCopy.Type == (DML_OPERATOR_TYPE) DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING)
                operatorDescCopy.Type = (DML_OPERATOR_TYPE) 169;
                
            // TODO: Change as new header is ingested
            if (operatorDescCopy.Type == (DML_OPERATOR_TYPE) DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT)
                operatorDescCopy.Type = (DML_OPERATOR_TYPE) 170;

            // Create and compile the operator.
            ComPtr<IDMLOperator> dmlOperator;
            ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&operatorDescCopy, IID_PPV_ARGS(&dmlOperator)));

            DML_EXECUTION_FLAGS executionFlags = GetExecutionFlags();
            ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(dmlOperator.Get(), executionFlags, IID_PPV_ARGS(&m_compiledOperator)));

            // Static buffer (might truncate name) to avoid excessive dynamic allocation only for debugging purposes.
            wchar_t nodeName[512];
            ORT_THROW_IF_FAILED(kernelInfo.GetNodeWrapperInterface()->GetWideName(sizeof(nodeName), nodeName));
            ORT_THROW_IF_FAILED(m_compiledOperator->SetName(nodeName));

            UINT64 persistentResourceSize = m_compiledOperator->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_executionProvider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Enabled,
                    m_persistentResource.GetAddressOf(),
                    m_persistentResourcePoolingUnk.GetAddressOf()));

                m_persistentResourceBinding = DML_BUFFER_BINDING{ m_persistentResource.Get(), 0, persistentResourceSize };
            }

            std::vector<DML_BUFFER_BINDING> initializationInputBindings(m_kernelInputIndices.size());

            ORT_THROW_IF_FAILED(m_executionProvider->InitializeOperator(
                m_compiledOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(initializationInputBindings)));
        }
    }

    void DmlOperator::SetDmlOperatorGraphDesc(
        const MLOperatorGraphDesc&& operatorGraphDesc,
        const MLOperatorKernelCreationContext& kernelInfo
        )
    {
        // Initialize should only be called once.
        assert(m_compiledOperator == nullptr);

        // DML doesn't support empty tensors. If an operator is still executable with empty tensors, the empty tensors
        // should be removed or massaged depending on the definition.
        for (const TensorDesc& desc : m_inputTensorDescs)
        {
            if (OperatorHelper::ContainsEmptyDimensions(desc.GetSizes()))
            {
                return;
            }
        }

        for (const TensorDesc& desc : m_outputTensorDescs)
        {
            if (OperatorHelper::ContainsEmptyDimensions(desc.GetSizes()))
            {
                return;
            }
        }

        // m_kernelInputIndices should be identity
        for (uint32_t idx = 0; idx < m_kernelInputIndices.size(); idx++)
        {
            if (m_kernelInputIndices[idx] == std::nullopt || !kernelInfo.IsInputValid(*m_kernelInputIndices[idx]))
            {
                continue;
            }
            assert(m_kernelInputIndices[idx] == idx);
        }

        // m_kernelOutputIndices should be identity
        for (uint32_t idx = 0; idx < m_kernelOutputIndices.size(); idx++)
        {
            if (m_kernelOutputIndices[idx] == std::nullopt || !kernelInfo.IsOutputValid(*m_kernelOutputIndices[idx]))
            {
                continue;
            }
            assert(m_kernelOutputIndices[idx] == idx);
        }

        ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));
        if (contextPrivate->IsDmlGraphNode())
        {
            ORT_THROW_IF_FAILED(contextPrivate->SetDmlOperator(&operatorGraphDesc));
        }
        else
        {
            DML_GRAPH_DESC graphDesc = {};
            std::vector<DML_GRAPH_NODE_DESC> dmlGraphNodes(operatorGraphDesc.nodeCount);
            std::vector<ComPtr<IDMLOperator>> dmlOperators(operatorGraphDesc.nodeCount);
            std::vector<DML_OPERATOR_GRAPH_NODE_DESC> dmlOperatorGraphNodes(operatorGraphDesc.nodeCount);
            std::vector<DML_GRAPH_EDGE_DESC> dmlInputEdges(operatorGraphDesc.inputEdgeCount);
            std::vector<DML_GRAPH_EDGE_DESC> dmlOutputEdges(operatorGraphDesc.outputEdgeCount);
            std::vector<DML_GRAPH_EDGE_DESC> dmlIntermediateEdges(operatorGraphDesc.intermediateEdgeCount);

            // DML Graph validator will check the validity of the graph. No need to check here.
            ConvertToDmlGraphDesc(operatorGraphDesc,
                                  graphDesc,
                                  dmlOperators,
                                  dmlOperatorGraphNodes,
                                  dmlGraphNodes,
                                  dmlInputEdges,
                                  dmlOutputEdges,
                                  dmlIntermediateEdges);

            // compile the graph and create IDMLCompiledOperator
            Microsoft::WRL::ComPtr<IDMLDevice1> dmlDevice1;
            DMLX_THROW_IF_FAILED(m_dmlDevice->QueryInterface(IID_PPV_ARGS(&dmlDevice1)));
            DML_EXECUTION_FLAGS executionFlags = GetExecutionFlags();
            ORT_THROW_IF_FAILED(dmlDevice1->CompileGraph(&graphDesc, executionFlags, IID_PPV_ARGS(&m_compiledOperator)));

            // Static buffer (might truncate name) to avoid excessive dynamic allocation only for debugging purposes.
            wchar_t nodeName[512];
            ORT_THROW_IF_FAILED(kernelInfo.GetNodeWrapperInterface()->GetWideName(sizeof(nodeName), nodeName));
            ORT_THROW_IF_FAILED(m_compiledOperator->SetName(nodeName));

            UINT64 persistentResourceSize = m_compiledOperator->GetBindingProperties().PersistentResourceSize;
            if (persistentResourceSize > 0)
            {
                ORT_THROW_IF_FAILED(m_executionProvider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Enabled,
                    m_persistentResource.GetAddressOf(),
                    m_persistentResourcePoolingUnk.GetAddressOf()));

                m_persistentResourceBinding = DML_BUFFER_BINDING{ m_persistentResource.Get(), 0, persistentResourceSize };
            }

            std::vector<DML_BUFFER_BINDING> initializationInputBindings(m_kernelInputIndices.size());

            ORT_THROW_IF_FAILED(m_executionProvider->InitializeOperator(
                m_compiledOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(initializationInputBindings)));
        }
    }

    void DmlOperator::SetDmlOperatorDesc(
        const DML_OPERATOR_DESC& operatorDesc,
        const MLOperatorKernelContext& kernelInfo
        )
    {
        // Create and compile the operator.
        // Unlike SetDmlOperatorDesc which takes a MLOperatorKernelCreationContext, it is okay to
        // call this method more than once, since Compute may take different inputs each execution.
        m_compiledOperator.Reset();
        ComPtr<IDMLOperator> dmlOperator;
        ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&operatorDesc, IID_PPV_ARGS(&dmlOperator)));
        ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(dmlOperator.Get(), GetExecutionFlags(), IID_PPV_ARGS(&m_compiledOperator)));

        UINT64 persistentResourceSize = m_compiledOperator->GetBindingProperties().PersistentResourceSize;
        if (persistentResourceSize > 0)
        {
            if (!m_persistentResource || m_persistentResource->GetDesc().Width < persistentResourceSize)
            {
                m_persistentResource = nullptr;
                ORT_THROW_IF_FAILED(m_executionProvider->AllocatePooledResource(
                    static_cast<size_t>(persistentResourceSize),
                    AllocatorRoundingMode::Enabled,
                    m_persistentResource.GetAddressOf(),
                    m_persistentResourcePoolingUnk.GetAddressOf()));
            }

            m_persistentResourceBinding = DML_BUFFER_BINDING{ m_persistentResource.Get(), 0, persistentResourceSize };
        }

        ORT_THROW_IF_FAILED(m_executionProvider->InitializeOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::span<const DML_BUFFER_BINDING>() // Empty input bindings since ownedByDml is not used.
            ));
    }

    void DmlOperator::Initialize(
        const MLOperatorKernelCreationContext& kernelInfo,
        uint32_t minDimensionCount
        )
    {
        Initialize(kernelInfo, std::nullopt, std::nullopt, std::nullopt, std::nullopt, minDimensionCount);
    }

    void DmlOperator::Initialize(
        const MLOperatorKernelCreationContext& kernelInfo,
        const std::optional<const std::vector<std::optional<uint32_t>>>& kernelInputIndices,
        const std::optional<const std::vector<std::optional<uint32_t>>>& kernelOutputIndices,
        const std::optional<gsl::span<const uint32_t>> inputShape,
        const std::optional<gsl::span<const uint32_t>> outputShape,
        uint32_t minDimensionCount
        )
    {
        if (kernelInputIndices)
        {
            m_kernelInputIndices = *kernelInputIndices;
        }
        else
        {
            m_kernelInputIndices.resize(kernelInfo.GetInputCount());
            std::iota(m_kernelInputIndices.begin(), m_kernelInputIndices.end(), 0);
        }

        if (kernelOutputIndices)
        {
            m_kernelOutputIndices = *kernelOutputIndices;
        }
        else
        {
            m_kernelOutputIndices.resize(kernelInfo.GetOutputCount());
            std::iota(m_kernelOutputIndices.begin(), m_kernelOutputIndices.end(), 0);
        }

        for (uint32_t i = 0; i < m_kernelInputIndices.size(); i++)
        {
            // Update m_kernelInputIndices to reflect optional tensors.
            if (m_kernelInputIndices[i] == std::nullopt ||
                !kernelInfo.IsInputValid(*m_kernelInputIndices[i]))
            {
                m_kernelInputIndices[i] = std::nullopt;
                m_inputTensorDescs.push_back(TensorDesc());
            }
            else
            {
                m_inputTensorDescs.push_back(CreateTensorDescFromInput(
                    kernelInfo,
                    *m_kernelInputIndices[i],
                    TensorAxis::DoNotCoerce,
                    TensorAxis::W,
                    TensorAxis::RightAligned,
                    inputShape,
                    minDimensionCount));
            }
        }

        for (uint32_t i = 0; i < m_kernelOutputIndices.size(); i++)
        {
            // Update m_kernelOutputIndices to reflect optional tensors.
            if (m_kernelOutputIndices[i] == std::nullopt ||
                !kernelInfo.IsOutputValid(*m_kernelOutputIndices[i]))
            {
                m_kernelOutputIndices[i] = std::nullopt;
                m_outputTensorDescs.push_back(TensorDesc());
            }
            else
            {
                m_outputTensorDescs.push_back(CreateTensorDescFromOutput(
                    kernelInfo,
                    *m_kernelOutputIndices[i],
                    TensorAxis::DoNotCoerce,
                    TensorAxis::W,
                    TensorAxis::RightAligned,
                    outputShape,
                    minDimensionCount));
            }
        }
    }

    void DmlOperator::InitializeInputsWithShapes(
        const MLOperatorKernelCreationContext& kernelInfo,
        const std::optional<const std::vector<std::optional<uint32_t>>>& kernelInputIndices,
        const std::optional<gsl::span<gsl::span<const uint32_t>>> inputShapes,
        uint32_t minDimensionCount)
    {
        if (kernelInputIndices)
        {
            m_kernelInputIndices = *kernelInputIndices;
        }
        else
        {
            m_kernelInputIndices.resize(kernelInfo.GetInputCount());
            std::iota(m_kernelInputIndices.begin(), m_kernelInputIndices.end(), 0);
        }


        for (uint32_t i = 0; i < m_kernelInputIndices.size(); i++)
        {
            // Update m_kernelInputIndices to reflect optional tensors.
            if (m_kernelInputIndices[i] == std::nullopt ||
                !kernelInfo.IsInputValid(*m_kernelInputIndices[i]))
            {
                m_kernelInputIndices[i] = std::nullopt;
                m_inputTensorDescs.push_back(TensorDesc());
            }
            else
            {
                auto edgeDesc = kernelInfo.GetInputEdgeDescription(*m_kernelInputIndices[i]);
                assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

                // prioritize the given input shapes
                TensorDesc tensorDesc;
                if (inputShapes.has_value() && i < (*inputShapes).size())
                {
                    tensorDesc = TensorDesc(
                        edgeDesc.tensorDataType,
                        (*inputShapes)[i], // desired
                        (*inputShapes)[i], // original
                        TensorAxis::DoNotCoerce,
                        TensorAxis::W,
                        TensorAxis::RightAligned,
                        minDimensionCount,
                        0
                    );
                }
                else if (kernelInfo.HasTensorShapeDescription())
                {
                    std::vector<uint32_t> actualTensorShape = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(*m_kernelInputIndices[i]);
                    tensorDesc = TensorDesc(
                        edgeDesc.tensorDataType,
                        actualTensorShape, // desired
                        actualTensorShape, // original
                        TensorAxis::DoNotCoerce,
                        TensorAxis::W,
                        TensorAxis::RightAligned,
                        minDimensionCount,
                        0
                    );
                }
                m_inputTensorDescs.push_back(tensorDesc);
            }
        }
    }

    void DmlOperator::InitializeOutputsWithShapes(
        const MLOperatorKernelCreationContext& kernelInfo,
        const std::optional<const std::vector<std::optional<uint32_t>>>& kernelOutputIndices,
        const std::optional<gsl::span<gsl::span<const uint32_t>>> outputShapes,
        uint32_t minDimensionCount)
    {
        if (kernelOutputIndices)
        {
            m_kernelOutputIndices = *kernelOutputIndices;
        }
        else
        {
            m_kernelOutputIndices.resize(kernelInfo.GetOutputCount());
            std::iota(m_kernelOutputIndices.begin(), m_kernelOutputIndices.end(), 0);
        }

        for (uint32_t i = 0; i < m_kernelOutputIndices.size(); i++)
        {
            // Update m_kernelOutputIndices to reflect optional tensors.
            if (m_kernelOutputIndices[i] == std::nullopt ||
                !kernelInfo.IsOutputValid(*m_kernelOutputIndices[i]))
            {
                m_kernelOutputIndices[i] = std::nullopt;
                m_outputTensorDescs.push_back(TensorDesc());
            }
            else
            {
                std::optional<gsl::span<const uint32_t>> outputShape;
                if (outputShapes.has_value() && i < (*outputShapes).size())
                {
                    outputShape = (*outputShapes)[i];
                }

                m_outputTensorDescs.push_back(CreateTensorDescFromOutput(
                    kernelInfo,
                    *m_kernelOutputIndices[i],
                    TensorAxis::DoNotCoerce,
                    TensorAxis::W,
                    TensorAxis::RightAligned,
                    outputShape,
                    minDimensionCount
                ));
            }
        }
    }

    void DmlOperator::InitializeWithShapes(
        const MLOperatorKernelCreationContext& kernelInfo,
        const std::optional<const std::vector<std::optional<uint32_t>>>& kernelInputIndices,
        const std::optional<const std::vector<std::optional<uint32_t>>>& kernelOutputIndices,
        const std::optional<gsl::span<gsl::span<const uint32_t>>> inputShapes,
        const std::optional<gsl::span<gsl::span<const uint32_t>>> outputShapes,
        uint32_t minDimensionCount
        )
    {
        InitializeInputsWithShapes(kernelInfo, kernelInputIndices, inputShapes, minDimensionCount);
        InitializeOutputsWithShapes(kernelInfo, kernelOutputIndices, outputShapes, minDimensionCount);
    }

    void DmlOperator::Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)));
    }

    bool DmlOperator::AllowHalfPrecisionComputation() const
    {
        // Most of our operators work with float data, but some do not. In those cases
        // no input params are float tensors. This function returns true if the operator
        // works with at least one float16 tensor and has no tensors of float32 type
        bool usesFloat16Tensors = false;

        for (const TensorDesc& desc : m_inputTensorDescs)
        {
            if (desc.GetDmlDataType() == DML_TENSOR_DATA_TYPE_FLOAT32)
            {
                return false;
            }

            if (desc.GetDmlDataType() == DML_TENSOR_DATA_TYPE_FLOAT16)
            {
                usesFloat16Tensors = true;
            }
        }

        for (const auto& desc : m_outputTensorDescs)
        {
            if (desc.GetDmlDataType() == DML_TENSOR_DATA_TYPE_FLOAT32)
            {
                return false;
            }
        }

        return usesFloat16Tensors;
    }

    DML_EXECUTION_FLAGS DmlOperator::GetExecutionFlags() const
    {
        DML_EXECUTION_FLAGS flags = DML_EXECUTION_FLAG_NONE;
        if (AllowHalfPrecisionComputation())
        {
            flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        }

        if (!m_executionProvider->MetacommandsEnabled())
        {
            flags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
        }

        return flags;
    }

    std::vector<IMLOperatorTensor*> DmlOperator::GetInputTensors(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors(m_kernelInputIndices.size());
        for (uint32_t i = 0; i < inputTensors.size(); i++)
        {
            if (m_kernelInputIndices[i] != std::nullopt)
            {
                assert(m_inputTensorDescs[i].IsValid());
                inputTensors[i] = kernelContext.GetInputTensor(*m_kernelInputIndices[i]).GetInterface().Get();
            }
        }

        return inputTensors;
    }

    std::vector<IMLOperatorTensor*> DmlOperator::GetOutputTensors(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> outputTensors(m_kernelOutputIndices.size());
        for (uint32_t i = 0; i < outputTensors.size(); i++)
        {
            if (m_kernelOutputIndices[i] != std::nullopt)
            {
                assert(m_outputTensorDescs[i].IsValid());
                outputTensors[i] = kernelContext.GetOutputTensor(*m_kernelOutputIndices[i]).GetInterface().Get();
            }
        }

        return outputTensors;
    }

    std::vector<IMLOperatorTensor*> DmlOperator::GetInputTensorsForExecute(const MLOperatorKernelContext& kernelContext)
    {
        return GetInputTensors(kernelContext);
    }

    std::vector<IMLOperatorTensor*> DmlOperator::GetOutputTensorsForExecute(const MLOperatorKernelContext& kernelContext)
    {
        return GetOutputTensors(kernelContext);
    }

    std::vector<DML_TENSOR_DESC> DmlOperator::GetDmlInputDescs()
    {
        std::vector<DML_TENSOR_DESC> descs(m_inputTensorDescs.size());
        for (size_t i = 0; i < descs.size(); i++)
        {
            descs[i] = m_inputTensorDescs[i].GetDmlDesc();
        }
        return descs;
    }

    std::vector<DML_TENSOR_DESC> DmlOperator::GetDmlOutputDescs()
    {
        std::vector<DML_TENSOR_DESC> descs(m_outputTensorDescs.size());
        for (size_t i = 0; i < descs.size(); i++)
        {
            descs[i] = m_outputTensorDescs[i].GetDmlDesc();
        }
        return descs;
    }

    ComPtr<IDMLCompiledOperator> DmlOperator::InitializeZeroInt64Tensor(uint64_t tensorSizeInBytes)
    {
        if (tensorSizeInBytes == 0)
        {
            return nullptr; // No work to do.
        }

        // This fun little solution uses DML's element-wise shader with XOR to zero the memory of the passed-in
        // tensor. This requires that the tensor's memory has been initialized (i.e. raw_mutable_data has been
        // called, and there is a size to the tensor). The tensor is XOR'd with itself to produce zeros,
        // and the operation is performed in-place on the same tensor.

        // Treat the tensor as a 1D array of 32-bit UINTs.
        uint32_t sizes[] = { 1, 1, 1, gsl::narrow<uint32_t>(tensorSizeInBytes / sizeof(uint32_t)) };

        DML_BUFFER_TENSOR_DESC bufferDesc = {};
        bufferDesc.DataType = DML_TENSOR_DATA_TYPE_UINT32;
        bufferDesc.Sizes = sizes;
        bufferDesc.DimensionCount = ARRAYSIZE(sizes);
        bufferDesc.TotalTensorSizeInBytes = tensorSizeInBytes;

        DML_TENSOR_DESC tensorDesc = { DML_TENSOR_TYPE_BUFFER, &bufferDesc };

        DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC xorDesc = {};
        xorDesc.ATensor = &tensorDesc;
        xorDesc.BTensor = &tensorDesc;
        xorDesc.OutputTensor = &tensorDesc;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR, &xorDesc };

        ComPtr<IDMLOperator> dmlOperator;
        ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(&dmlOperator)));

        ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
        ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(dmlOperator.Get(), GetExecutionFlags(), IID_PPV_ARGS(&dmlCompiledOperator)));

        return dmlCompiledOperator;
    }

    void DmlOperator::ExecuteZeroInt64Tensor(IDMLCompiledOperator* compiledOperator, IMLOperatorTensor* tensor)
    {
        // Element-wise XOR takes two inputs and an output. We want in-place execution, so all three
        // resources are the same.
        IMLOperatorTensor* inputTensors[] = { tensor, tensor };
        IMLOperatorTensor* outputTensors[] = { tensor };

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            compiledOperator,
            nullptr, // persistent resource binding
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)
            ));
    }

    TensorDesc DmlOperator::CreateTensorDescFromInput(
        const MLOperatorKernelCreationContext& kernelInfo,
        uint32_t index,
        int32_t coerceAxis,
        int32_t placement,
        int32_t leftAlignedDimensionCount,
        std::optional<gsl::span<const uint32_t>> tensorShape,
        uint32_t minDimensionCount
        ) const
    {
        if (!kernelInfo.IsInputValid(index))
        {
            // The tensor is optional.
            return TensorDesc();
        }

        auto edgeDesc = kernelInfo.GetInputEdgeDescription(index);
        assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

        std::vector<uint32_t> actualTensorShape;
        if (kernelInfo.HasTensorShapeDescription())
        {
            actualTensorShape = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(index);
        }
        else
        {
            // The tensor has delayed shape determination.
            return TensorDesc();
        }

        return TensorDesc(
            edgeDesc.tensorDataType,
            tensorShape ? *tensorShape : actualTensorShape,
            actualTensorShape,
            coerceAxis,
            placement,
            leftAlignedDimensionCount,
            minDimensionCount,
            0
            );
    }

    TensorSequenceDesc DmlOperator::CreateTensorSequenceDescFromInput(
        const MLOperatorKernelCreationContext& kernelInfo,
        uint32_t index,
        int32_t coerceAxis,
        int32_t placement,
        int32_t leftAlignedDimensionCount,
        std::optional<gsl::span<const uint32_t>> tensorShape,
        uint32_t minDimensionCount
        ) const
    {
        if (!kernelInfo.IsInputValid(index))
        {
            // The tensor is optional.
            return TensorSequenceDesc();
        }

        auto edgeDesc = kernelInfo.GetInputEdgeDescription(index);
        assert(edgeDesc.edgeType == MLOperatorEdgeType::SequenceTensor);
        ORT_THROW_HR_IF(E_INVALIDARG, edgeDesc.edgeType != MLOperatorEdgeType::SequenceTensor);

        const auto& shapeDescription = kernelInfo.GetTensorShapeDescription();
        const uint32_t numTensors = shapeDescription.GetSequenceInputCount(index);

        TensorSequenceDesc tensorDescs;
        tensorDescs.reserve(numTensors);

        for (uint32_t sequenceIndex = 0; sequenceIndex < numTensors; ++sequenceIndex)
        {
            std::vector<uint32_t> actualTensorShape;
            if (kernelInfo.HasTensorShapeDescription())
            {
                actualTensorShape = shapeDescription.GetSequenceInputTensorShape(index, sequenceIndex);

                tensorDescs.emplace_back(
                    edgeDesc.tensorDataType,
                    tensorShape ? *tensorShape : actualTensorShape,
                    actualTensorShape,
                    coerceAxis,
                    placement,
                    leftAlignedDimensionCount,
                    minDimensionCount,
                    0);
            }
            else
            {
                // The tensor has delayed shape determination.
                tensorDescs.push_back(TensorDesc());
            }
        }

        return tensorDescs;
    }

    TensorDesc DmlOperator::CreateTensorDescFromOutput(
        const MLOperatorKernelCreationContext& kernelInfo,
        uint32_t index,
        int32_t coerceAxis,
        int32_t placement,
        int32_t leftAlignedDimensionCount,
        std::optional<gsl::span<const uint32_t>> tensorShape,
        uint32_t minDimensionCount
        ) const
    {
        if (!kernelInfo.IsOutputValid(index))
        {
            // The tensor is optional.
            return TensorDesc();
        }

        auto edgeDesc = kernelInfo.GetOutputEdgeDescription(index);
        assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

        if (!kernelInfo.HasTensorShapeDescription())
        {
            // The tensor has delayed shape determination.
            return TensorDesc(edgeDesc.tensorDataType);
        }

        MLOperatorTensorShapeDescription outputShapeDescription = kernelInfo.GetTensorShapeDescription();
        if (!outputShapeDescription.HasOutputShapeDescription())
        {
            // The tensor has delayed shape determination.
            return TensorDesc();
        }

        auto outputShape = outputShapeDescription.GetOutputTensorShape(index);

        return TensorDesc(
            edgeDesc.tensorDataType,
            tensorShape ? *tensorShape : outputShape,
            tensorShape ? *tensorShape : outputShape,
            coerceAxis,
            placement,
            leftAlignedDimensionCount,
            minDimensionCount,
            0
            );
    }

    void DmlOperator::ConvertToDmlGraphDesc(const MLOperatorGraphDesc& operatorGraphDesc,
                                            _Out_ DML_GRAPH_DESC& graphDesc,
                                            _Inout_ std::vector<ComPtr<IDMLOperator>>& dmlOperators,
                                            _Inout_ std::vector<DML_OPERATOR_GRAPH_NODE_DESC>& dmlOperatorGraphNodes,
                                            _Inout_ std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
                                            _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
                                            _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
                                            _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges)
    {
        graphDesc.InputCount = gsl::narrow_cast<uint32_t>(m_kernelInputIndices.size());
        graphDesc.OutputCount = gsl::narrow_cast<uint32_t>(m_kernelOutputIndices.size());

        // set the graph nodes
        graphDesc.NodeCount = operatorGraphDesc.nodeCount;
        for (size_t i = 0; i < graphDesc.NodeCount; ++i)
        {
            DML_OPERATOR_DESC opDesc = *operatorGraphDesc.nodesAsOpDesc[i];
            
            // TODO: Change as new header is ingested
            if (opDesc.Type == (DML_OPERATOR_TYPE) DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING)
                opDesc.Type = (DML_OPERATOR_TYPE) 169;
                
            // TODO: Change as new header is ingested
            if (opDesc.Type == (DML_OPERATOR_TYPE) DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT)
                opDesc.Type = (DML_OPERATOR_TYPE) 170;

            // Create the operator.
            ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(&dmlOperators[i])));
            dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{dmlOperators[i].Get()};
            dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]};
        }
        graphDesc.Nodes = dmlGraphNodes.data();

        // set the input edges
        graphDesc.InputEdgeCount = operatorGraphDesc.inputEdgeCount;
        for (size_t i = 0; i < operatorGraphDesc.inputEdgeCount; ++i)
        {
            dmlInputEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &operatorGraphDesc.inputEdges[i]};
        }
        graphDesc.InputEdges = dmlInputEdges.data();

        // set the output edges
        graphDesc.OutputEdgeCount = operatorGraphDesc.outputEdgeCount;
        for (size_t i = 0; i < operatorGraphDesc.outputEdgeCount; ++i)
        {
            dmlOutputEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &operatorGraphDesc.outputEdges[i]};
        }
        graphDesc.OutputEdges = dmlOutputEdges.data();

        // set the intermediate edges
        graphDesc.IntermediateEdgeCount = operatorGraphDesc.intermediateEdgeCount;
        for (size_t i = 0; i < operatorGraphDesc.intermediateEdgeCount; ++i)
        {
            dmlIntermediateEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &operatorGraphDesc.intermediateEdges[i]};
        }
        graphDesc.IntermediateEdges = dmlIntermediateEdges.data();
    }

    /*static*/ void DmlOperator::TryConvertTensorToBroadcastScalar(
        const MLOperatorKernelCreationContext& kernelInfo, 
        const DML_TENSOR_DESC* tensor, 
        uint32_t kernelInputIndex)
    {
        if (!tensor)
        {
            return;
        }

        auto constExpTensor = kernelInfo.TryGetConstantInputTensor(kernelInputIndex);
        if (!constExpTensor)
        {
            return;
        }
        else if (!IsCpuData())
        {
            return;
        }

        uint32_t totalKernelInputElementCount = constExpTensor->GetTotalElementCount();
        if (totalKernelInputElementCount <= 1)
        {
            return;
        }        
        
        uint32_t elementSize = 0;

        switch (constExpTensor->GetTensorDataType())
        {
        case MLOperatorTensorDataType::UInt8:
        case MLOperatorTensorDataType::Int8:
            elementSize = 1;
            break;

        case MLOperatorTensorDataType::Float16:
        case MLOperatorTensorDataType::UInt16:
        case MLOperatorTensorDataType::Int16:
            elementSize = 2;
            break;
            
        case MLOperatorTensorDataType::/*Float32*/Float:
        case MLOperatorTensorDataType::UInt32:
        case MLOperatorTensorDataType::Int32:
            elementSize = 4;
            break;

        case MLOperatorTensorDataType::/*Float64*/Double:
        case MLOperatorTensorDataType::UInt64:
        case MLOperatorTensorDataType::Int64:
            elementSize = 8;
            break;

        default:
            return;
        }

        const std::uint8_t* byteData = static_cast<const std::uint8_t*>(constExpTensor->GetByteData());

        assert(tensor->Type == DML_TENSOR_TYPE_BUFFER);
        auto *bufferTensorDesc = const_cast<DML_BUFFER_TENSOR_DESC*>(static_cast<const DML_BUFFER_TENSOR_DESC*>(tensor->Desc));

        for (size_t i = 1; i < totalKernelInputElementCount; ++i)
        {
            if (memcmp(byteData, byteData + i * elementSize, elementSize))
            {
                return;
            }
        }

        if (bufferTensorDesc->DimensionCount > sizeof(zeroArray) / sizeof(zeroArray[0]))
        {
            assert(false);
            return;
        }

        bufferTensorDesc->Strides = zeroArray;
        bufferTensorDesc->TotalTensorSizeInBytes = (elementSize + 3) & ~3;
    }

} // namespace Dml
