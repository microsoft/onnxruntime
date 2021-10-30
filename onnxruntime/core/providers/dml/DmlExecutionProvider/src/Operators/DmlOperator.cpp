// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
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

        // Create and compile the operator.
        ComPtr<IDMLOperator> dmlOperator;
        ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&operatorDesc, IID_PPV_ARGS(&dmlOperator)));

        ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));

        if (contextPrivate->IsDmlGraphNode())
        {
            // Create an edge list using sentinels for unused edges, as required by the SetDmlOperator ABI
            auto ReplaceUnusedEdgeIndicesWithSentinel = [](gsl::span<const std::optional<uint32_t>> indices)
            {
                std::vector<uint32_t> ret;
                ret.reserve(indices.size());
                for (const std::optional<uint32_t>& index : indices)
                {
                    ret.push_back(index.has_value() ? index.value() : std::numeric_limits<uint32_t>::max());
                }

                return ret;
            };

            MLOperatorKernelDmlProperties properties = {};
            auto kernelInputIndices = ReplaceUnusedEdgeIndicesWithSentinel(m_kernelInputIndices);
            properties.dmlInputCount = static_cast<uint32_t>(kernelInputIndices.size());
            properties.kernelInputIndices = kernelInputIndices.data();
            
            auto kernelOutputIndices = ReplaceUnusedEdgeIndicesWithSentinel(m_kernelOutputIndices);
            properties.dmlOutputCount = static_cast<uint32_t>(kernelOutputIndices.size());
            properties.kernelOutputIndices = kernelOutputIndices.data();
            properties.allowHalfPrecisionComputation = AllowHalfPrecisionComputation();

            ORT_THROW_IF_FAILED(contextPrivate->SetDmlOperator(dmlOperator.Get(), &operatorDesc, &properties));
        }
        else
        {
            DML_EXECUTION_FLAGS executionFlags = GetExecutionFlags();
            ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(dmlOperator.Get(), executionFlags, IID_PPV_ARGS(&m_compiledOperator)));

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

    void DmlOperator::Remap64bitDmlDataTypesTo32bit()
    {
        for (auto& tensor : m_inputTensorDescs)
        {
            tensor.Remap64bitDmlDataTypeTo32bit();
        }

        for (auto& tensor : m_outputTensorDescs)
        {
            tensor.Remap64bitDmlDataTypeTo32bit();
        }
    }

    void DmlOperator::Remap64bitDmlDataTypesTo32bitIfNeeded()
    {
        // Conditionally remap 64-bit data types to strided 32-bit if DML does not
        // support 64-bit data types directly on the device.

        uint32_t deviceTypeMask = Dml::GetSupportedDeviceDataTypeMask(m_dmlDevice.Get());
        uint32_t deviceTypeMask64bit = (1 << DML_TENSOR_DATA_TYPE_INT64) | (1 << DML_TENSOR_DATA_TYPE_UINT64);

        // If the device doesn't support 64-bit tensors, fall back to 32-bit with strides.
        if (!(deviceTypeMask & deviceTypeMask64bit))
        {
            Remap64bitDmlDataTypesTo32bit();
        }
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

} // namespace Dml