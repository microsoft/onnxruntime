// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"
#include "core/providers/dml/DmlExecutionProvider/src/ExecutionProvider.h"

namespace Dml
{

class DmlOperatorNonZero: public DmlOperator
{
public:
    DmlOperatorNonZero(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        std::vector<DimensionType> inputShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);

        // Scalars have a rank of 0, but DML only supports 1 and more, which is the same
        if (inputShape.empty())
        {
            inputShape.push_back(1);
        }

        uint32_t numElements = ComputeElementCountFromDimensions(inputShape);

        gsl::span<const uint32_t> inputShapes[1] = {inputShape};
        DmlOperator::InitializeWithShapes(kernelCreationContext, std::nullopt, std::nullopt, inputShapes, std::nullopt, 1);

        m_rank = static_cast<DimensionType>(inputShape.size());
        std::vector<DimensionType> outputCountShape = {1};
        std::vector<DimensionType> outputCoordinatesShape = {numElements, m_rank};

        // TODO: Remove the doubled strides when DML supports native int64 for NonZero
        // TensorFlow outputs {rank, numElements}, but DML outputs {numElements, rank}
        std::vector<DimensionType> outputCoordinatesStrides = {2, numElements * 2};
        m_intermediateTensorDescs = {
            TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, outputCountShape),
            TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, outputCoordinatesShape, outputCoordinatesStrides),
        };

        // If the input has no elements, bypass the DML execution
        if (numElements == 0)
        {
            m_emptyInput = true;
        }
        else
        {
            m_emptyInput = false;
            m_outputCountShape = {1};
            m_outputCoordinatesShape = {static_cast<int64_t>(numElements), static_cast<int64_t>(m_rank)};

            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> intermediateDescs(m_intermediateTensorDescs.size());
            for (size_t i = 0; i < intermediateDescs.size(); i++)
            {
                intermediateDescs[i] = m_intermediateTensorDescs[i].GetDmlDesc();
            }

            DML_NONZERO_COORDINATES_OPERATOR_DESC nonzeroCoordinatesDesc = {};
            nonzeroCoordinatesDesc.InputTensor = &inputDescs[0];
            nonzeroCoordinatesDesc.OutputCountTensor = &intermediateDescs[0];
            nonzeroCoordinatesDesc.OutputCoordinatesTensor = &intermediateDescs[1];

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_NONZERO_COORDINATES, &nonzeroCoordinatesDesc };
            SetDmlOperatorDesc(opDesc, kernelCreationContext);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        ExecutionProviderImpl* executionProvider = static_cast<ExecutionProviderImpl*>(m_executionProvider.Get());

        // Create the DML output tensor for the number of nonzero elements
        onnxruntime::Tensor outputCountDml(onnxruntime::DataTypeImpl::GetType<uint32_t>(), m_outputCountShape, executionProvider->GetGpuAllocator());
        Microsoft::WRL::ComPtr<IMLOperatorTensor> outputCountDmlWrapper = wil::MakeOrThrow<Windows::AI::MachineLearning::Adapter::TensorWrapper>(
            &outputCountDml,
            true,
            executionProvider,
            true);

        // Create the DML output tensor for the coordinates (not cropped)
        onnxruntime::Tensor intermediateCoordinatesDml(onnxruntime::DataTypeImpl::GetType<int64_t>(), m_outputCoordinatesShape, executionProvider->GetGpuAllocator());
        Microsoft::WRL::ComPtr<IMLOperatorTensor> intermediateCoordinatesDmlWrapper = wil::MakeOrThrow<Windows::AI::MachineLearning::Adapter::TensorWrapper>(
            &intermediateCoordinatesDml,
            true,
            executionProvider,
            true);

        std::vector<IMLOperatorTensor*> nonzeroCoordinatesInputTensors = GetInputTensors(kernelContext);
        std::vector<IMLOperatorTensor*> nonzeroCoordinatesOutputTensors = {outputCountDmlWrapper.Get(), intermediateCoordinatesDmlWrapper.Get()};

        uint32_t nonzeroElementCount = 0;

        if (!m_emptyInput)
        {
            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                m_compiledOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(nonzeroCoordinatesInputTensors),
                gsl::make_span(nonzeroCoordinatesOutputTensors)));

            // Copy the number of nonzero elements back to the CPU
            onnxruntime::Tensor outputCountCpu(onnxruntime::DataTypeImpl::GetType<uint32_t>(), {1}, executionProvider->GetCpuInputAllocator());
            Microsoft::WRL::ComPtr<IMLOperatorTensor> outputCountCpuWrapper = wil::MakeOrThrow<Windows::AI::MachineLearning::Adapter::TensorWrapper>(
                &outputCountCpu,
                false,
                executionProvider,
                true);
            ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
                outputCountCpuWrapper.Get(),
                nonzeroCoordinatesOutputTensors.front()));
            nonzeroElementCount = outputCountCpu.Data<uint32_t>()[0];
        }

        // Create the final output tensor, which is cropped to the actual number of nonzero elements
        std::vector<uint32_t> outputSizes({m_rank, nonzeroElementCount});
        auto outputTensor = kernelContext.GetOutputTensor(0, outputSizes);

        if (!m_emptyInput && nonzeroElementCount > 0)
        {
            // TODO: Remove this hack when DML supports native int64 for NonZero
            // We use the int64/uint32 stride hack here, so zero out the data before writing to it
            uint64_t tensorSizeInBytes = uint64_t(m_rank) * uint64_t(nonzeroElementCount) * sizeof(int64_t);
            ComPtr<IDMLCompiledOperator> zeroOperator = InitializeZeroInt64Tensor(tensorSizeInBytes);

            // TODO: Remove this hack when DML supports native int64 for NonZero
            ExecuteZeroInt64Tensor(zeroOperator.Get(), outputTensor.GetInterface().Get());

            ComPtr<IDMLCompiledOperator> sliceOperator = InitializeSlice(m_intermediateTensorDescs[1], nonzeroElementCount);

            // Finally, we crop the output to the actual number of nonzero elements, thus removing the padding
            std::array<IMLOperatorTensor*, 1> sliceInputTensors = {nonzeroCoordinatesOutputTensors[1]};
            std::array<IMLOperatorTensor*, 1> sliceOutputTensors = {outputTensor.GetInterface().Get()};

            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                sliceOperator.Get(),
                nullptr, // persistent resource binding
                sliceInputTensors,
                sliceOutputTensors));
        }
    }

private:
    ComPtr<IDMLCompiledOperator> InitializeSlice(TensorDesc& inputDesc, uint32_t nonzeroElementCount)
    {
        assert(inputDesc.GetSizes().size() == 2);

        uint32_t rank = inputDesc.GetSizes().back();
        std::array<uint32_t, 2> inputWindowOffsets = {0, 0};
        std::array<int32_t, 2> inputWindowStrides = {1, 1};
        std::array<uint32_t, 2> inputWindowSizes = {nonzeroElementCount, rank};

        // TODO: Remove the doubled strides when DML supports native int64 for NonZero
        std::array<uint32_t, 2> outputStrides = {2, nonzeroElementCount * 2};
        TensorDesc outputDesc(inputDesc.GetDmlDataType(), inputWindowSizes, outputStrides);

        const auto inputOpDesc = inputDesc.GetDmlDesc();
        const auto outputOpDesc = outputDesc.GetDmlDesc();

        DML_SLICE1_OPERATOR_DESC sliceDesc = {};
        sliceDesc.DimensionCount = 2;
        sliceDesc.InputWindowOffsets = inputWindowOffsets.data();
        sliceDesc.InputWindowSizes = inputWindowSizes.data();
        sliceDesc.InputWindowStrides = inputWindowStrides.data();
        sliceDesc.InputTensor = &inputOpDesc;
        sliceDesc.OutputTensor = &outputOpDesc;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SLICE1, &sliceDesc };

        ComPtr<IDMLOperator> dmlOperator;
        ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&opDesc, IID_PPV_ARGS(&dmlOperator)));

        ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
        ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(dmlOperator.Get(), GetExecutionFlags(), IID_PPV_ARGS(&dmlCompiledOperator)));

        return dmlCompiledOperator;
    }

    std::vector<TensorDesc> m_intermediateTensorDescs;
    onnxruntime::TensorShape m_outputCountShape;
    onnxruntime::TensorShape m_outputCoordinatesShape;
    bool m_emptyInput = false;
    uint32_t m_rank = 0;
};

DML_OP_DEFINE_CREATION_FUNCTION(NonZero, DmlOperatorNonZero);

} // namespace Dml
