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
    std::vector<TensorDesc> m_inputTensorDescs;
    std::vector<uint32_t> m_inputIndices;
    TensorDesc m_outputTensorDesc;
    Shape m_outputShape;

public:

    DmlOperatorConcatFromSequence(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo)
    {
        // Ensure there is only 1 input, and 1 output
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        // Ensure there the singular input is a squence of tensors
        auto edgeDesc = kernelInfo.GetInputEdgeDescription(0);
        assert(edgeDesc.edgeType == MLOperatorEdgeType::SequenceTensor);
        auto sequenceInputDataType = edgeDesc.tensorDataType;
        auto sequenceInputDmlDataType = Dml::GetDmlDataTypeFromMlDataTypeNoThrow(sequenceInputDataType);

        // Ensure there the singular output is a tensors        
        edgeDesc = kernelInfo.GetOutputEdgeDescription(0);
        assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

        // Get the number of tensors in the sequence of tensors input.
        // If no tensors are in the sequence, then no evaluation should occur.
        // In this case the output shape is default initialized to {}.
        auto tensorShapeDescription = kernelInfo.GetTensorShapeDescription();
        auto numTensorsInSequence = tensorShapeDescription.GetSequenceInputCount(0);
        if (numTensorsInSequence > 0)
        {
            uint32_t axis = 0;
            uint32_t axisTotal = 0;
            std::optional<uint32_t> inputDimCount;
            for (uint32_t i = 0; i < numTensorsInSequence; i++)
            {
                // Remove empty tensors and keep only the non-empty tensors
                auto shape = tensorShapeDescription.GetSequenceInputTensorShape(0, i);
                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    continue;
                }

                // When processing the first tensor, initialize output shape, inputDimCount and axis.
                if (!inputDimCount)
                {
                    m_outputShape = shape;
                    inputDimCount = static_cast<uint32_t>(shape.size());
                    axis = static_cast<uint32_t>(HandleNegativeAxis(kernelInfo.GetOptionalAttribute<int>(AttrName::Axis, -1), *inputDimCount));
                    ML_CHECK_VALID_ARGUMENT(axis < inputDimCount);
                }
                
                ML_CHECK_BOOL(*inputDimCount == shape.size());
                m_inputTensorDescs.emplace_back(TensorDesc(sequenceInputDmlDataType, shape));
                m_inputIndices.push_back(i);
                axisTotal += shape[axis];
            }

            // We should only call join if there exists input tensors that are non-empty and non-scalar.
            // In that case, the inputDimCount must be set and greater than 0.
            if (inputDimCount.has_value() && *inputDimCount > 0)
            {
                m_outputShape[axis] = axisTotal;
                m_outputTensorDesc = TensorDesc(sequenceInputDmlDataType, m_outputShape);
                auto dmlAxis = GetDmlAdjustedAxis(axis, *inputDimCount, m_outputTensorDesc.GetDimensionCount());

                auto outputIndices = std::vector<std::optional<uint32_t>> { 0 };
                gsl::span<const uint32_t> outputShapes[1] = { m_outputShape };
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
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        if (!m_inputIndices.size())
        {
            return;
        }

        auto operatorKernelContext = kernelContext.GetInterface();
        auto inputTensors = std::vector<IMLOperatorTensor*>(m_inputIndices.size());
        auto tensors = std::vector<ComPtr<IMLOperatorTensor>>(m_inputIndices.size());
        for (uint32_t i = 0; i < inputTensors.size(); i++)
        {
            assert(m_inputTensorDescs[i].IsValid());
            ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceInputTensor(0, m_inputIndices[i], tensors[i].GetAddressOf()));
            inputTensors[i] = tensors[i].Get();
        }

        auto outputTensor = kernelContext.GetOutputTensor(0, m_outputShape).GetInterface().Get();
        auto outputTensors = gsl::span<IMLOperatorTensor*> { &outputTensor, 1 };

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            outputTensors));
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ConcatFromSequence, DmlOperatorConcatFromSequence);

} // namespace Dml
