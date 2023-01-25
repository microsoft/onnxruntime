// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorConcatFromSequence : public DmlOperator
{
public:
    using Shape = std::vector<DimensionType>;
    using Self = DmlOperatorConcatFromSequence;

private:
    int m_axis = 0;
    std::vector<gsl::span<const uint32_t>> m_inputShapes;
    std::vector<TensorDesc> m_inputTensorDescs;
    std::vector<uint32_t> m_inputIndices;
    TensorDesc m_outputTensorDesc;
    Shape m_outputShape;

public:

    DmlOperatorConcatFromSequence(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        auto tensorShapeDescription = kernelInfo.GetTensorShapeDescription();
        auto numTensorsInSequence = tensorShapeDescription.GetSequenceInputCount(0);
        if (numTensorsInSequence > 0)
        {
            uint32_t axisTotal = 0;
            std::optional<uint32_t> inputDimCount;
            for (uint32_t i = 0; i < numTensorsInSequence; i++)
            {
                // Only keep the non-empty tensors
                auto shape = tensorShapeDescription.GetSequenceInputTensorShape(0, i);
                if (!OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    if (!inputDimCount)
                    {
                        inputDimCount = static_cast<uint32_t>(shape.size());
                        m_outputShape = shape;
                        m_axis = static_cast<int>(HandleNegativeAxis(kernelInfo.GetOptionalAttribute<int>(AttrName::Axis, -1), *inputDimCount));
                        ML_CHECK_VALID_ARGUMENT(m_axis < static_cast<int>(shape.size()));
                    }
                    else
                    {
                        ML_CHECK_BOOL(*inputDimCount == shape.size());
                    }

                    axisTotal += shape[m_axis];
                    m_inputTensorDescs.emplace_back(TensorDesc::ConstructDefaultTensorDesc(MLOperatorTensorDataType::Float, shape));
                    m_inputShapes.emplace_back(std::move(shape));
                    m_inputIndices.push_back(i);
                }
            }

            m_outputShape[m_axis] = axisTotal;
            m_outputTensorDesc = TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_outputShape); // TODO fix data type
            uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, *inputDimCount, m_outputTensorDesc.GetDimensionCount());

            std::vector<std::optional<uint32_t>> outputIndices = { 0 };
            gsl::span<const uint32_t> outputShapes[1] = {m_outputShape};
            DmlOperator::InitializeOutputsWithShapes(kernelInfo, outputIndices, outputShapes, 1);

            auto outputDescs = std::vector<DML_TENSOR_DESC> { m_outputTensorDesc.GetDmlDesc() };
            auto inputDescs = std::vector<DML_TENSOR_DESC>(m_inputTensorDescs.size());

            for (int i = 0; i < inputDescs.size(); i++)
            {
                inputDescs[i] = m_inputTensorDescs[i].GetDmlDesc();
            }

            DML_JOIN_OPERATOR_DESC joinDesc = {};
            joinDesc.InputCount = gsl::narrow_cast<uint32_t>(inputDescs.size());
            joinDesc.InputTensors = inputDescs.data();
            joinDesc.OutputTensor = outputDescs.data();
            joinDesc.Axis = dmlAxis;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_JOIN, &joinDesc };

            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        auto operatorKernelContext = kernelContext.GetInterface();
        std::vector<IMLOperatorTensor*> inputTensors(m_inputIndices.size());
        std::vector<ComPtr<IMLOperatorTensor>> tensors(m_inputIndices.size());
        for (uint32_t i = 0; i < inputTensors.size(); i++)
        {
            assert(m_inputTensorDescs[i].IsValid());
            ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceInputTensor(0, m_inputIndices[i], tensors[i].GetAddressOf()));
            inputTensors[i] = tensors[i].Get();
        }

        IMLOperatorTensor* outputTensor = kernelContext.GetOutputTensor(0, m_outputShape).GetInterface().Get();
        gsl::span<IMLOperatorTensor*> outputTensors{ &outputTensor, 1 };

        if (!inputTensors.empty())
        {
            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                m_compiledOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(inputTensors),
                outputTensors));
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ConcatFromSequence, DmlOperatorConcatFromSequence);

} // namespace Dml
