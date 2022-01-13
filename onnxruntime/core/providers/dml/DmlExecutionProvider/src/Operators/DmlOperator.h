// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "OperatorUtility.h"

namespace Dml
{
    class DmlOperator
    {
    public:
        DmlOperator(const MLOperatorKernelCreationContext& kernelInfo);

        virtual ~DmlOperator() = default;

        virtual void Compute(const MLOperatorKernelContext& kernelContext);

    protected:
        ComPtr<IExecutionProvider> m_executionProvider; 
        ComPtr<IDMLDevice> m_dmlDevice;

        // Tensor descs ordered based on index arrays passed to Initialize
        std::vector<TensorDesc> m_inputTensorDescs;
        std::vector<TensorDesc> m_outputTensorDescs;

        ComPtr<IDMLCompiledOperator> m_compiledOperator;
        ComPtr<ID3D12Resource> m_persistentResource;
        ComPtr<IUnknown> m_persistentResourcePoolingUnk; // Controls when the persistent resource is returned to the pool
        std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;

        void Initialize(
            const MLOperatorKernelCreationContext& kernelInfo,
            uint32_t minDimensionCount
            );

        void Initialize(
            const MLOperatorKernelCreationContext& kernelInfo,
            const std::optional<const std::vector<std::optional<uint32_t>>>& kernelInputIndices = std::nullopt,
            const std::optional<const std::vector<std::optional<uint32_t>>>& kernelOutputIndices = std::nullopt,
            const std::optional<gsl::span<const uint32_t>> inputShape = std::nullopt,
            const std::optional<gsl::span<const uint32_t>> outputShape = std::nullopt,
            uint32_t minDimensionCount = NchwDimensionCount
            );

        bool AllowHalfPrecisionComputation() const;
        DML_EXECUTION_FLAGS GetExecutionFlags() const;

        void SetDmlOperatorDesc(
            const DML_OPERATOR_DESC& operatorDesc, 
            const MLOperatorKernelCreationContext& kernelInfo
            );

        void SetDmlOperatorDesc(
            const DML_OPERATOR_DESC& operatorDesc,
            const MLOperatorKernelContext& kernelInfo
            );
        
        // Tensors ordered based on index arrays passed to Initialize
        std::vector<IMLOperatorTensor*> GetInputTensors(const MLOperatorKernelContext& kernelContext);
        std::vector<IMLOperatorTensor*> GetOutputTensors(const MLOperatorKernelContext& kernelContext);
        
        // Retrieves the input/output tensors to be supplied to DirectML for execution. These differ from
        // Get[Input|Output]Tensors in that they account for the binding requirements of DML, instead of
        // unconditionally retrieving all input and output tensors.
        std::vector<IMLOperatorTensor*> GetInputTensorsForExecute(const MLOperatorKernelContext& kernelContext);
        std::vector<IMLOperatorTensor*> GetOutputTensorsForExecute(const MLOperatorKernelContext& kernelContext);

        // Tensor descs ordered based on index arrays passed to Initialize
        std::vector<DML_TENSOR_DESC> GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> GetDmlOutputDescs();

        // Sets the memory of a tensor to all zeros.
        //
        // WinML requires int64_t for certain operators, like ArgMax and ArgMin. DML does not directly support
        // int64_t as a tensor data type, because D3D does not support 64-bit integers. Currently, we "hack"
        // support for int64_t WinML tensors using int32_t tensors with strides; the upper 32-bits are not used,
        // since this hack is only used for unsigned values that require less than 32 bits. However, WinML
        // will read the full 64-bit values. This means it is necessary to zero out the memory to ensure there
        // are no uninitialized values in the upper 32-bit portion of the tensor memory.
        //
        // It returns nullptr if there is no work to do (0 bytes).
        //
        ComPtr<IDMLCompiledOperator> InitializeZeroInt64Tensor(uint64_t tensorSizeInBytes);

        void ExecuteZeroInt64Tensor(IDMLCompiledOperator* compiledOperator, IMLOperatorTensor* tensor);

        TensorDesc CreateTensorDescFromInput(
            const MLOperatorKernelCreationContext& kernelInfo,
            uint32_t index,
            int32_t coerceAxis = TensorAxis::DoNotCoerce,
            int32_t placement = TensorAxis::W,
            int32_t leftAlignedDimensionCount = TensorAxis::RightAligned,
            std::optional<gsl::span<const uint32_t>> tensorShape = std::nullopt,
            uint32_t minDimensionCount = NchwDimensionCount
            ) const;

        TensorDesc CreateTensorDescFromOutput(
            const MLOperatorKernelCreationContext& kernelInfo,
            uint32_t index,
            int32_t coerceAxis = TensorAxis::DoNotCoerce,
            int32_t placement = TensorAxis::W,
            int32_t leftAlignedDimensionCount = TensorAxis::RightAligned,
            std::optional<gsl::span<const uint32_t>> tensorShape = std::nullopt,
            uint32_t minDimensionCount = NchwDimensionCount
            ) const;

    private:
        // For each input or output of the DML kernel, the corresponding input or output of the original 
        // kernel.  Entries for unused DML inputs are nullopt.
        std::vector<std::optional<uint32_t>> m_kernelInputIndices;
        std::vector<std::optional<uint32_t>> m_kernelOutputIndices;
    };

} // namespace Dml
