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
        auto new_axis = static_cast<uint32_t>(kernelInfo.GetOptionalAttribute<int>(AttrName::NewAxis, 0));
        ML_CHECK_VALID_ARGUMENT(1 == new_axis || 0 == new_axis);

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

                auto r = static_cast<uint32_t>(shape.size());
                auto is_scalar = r == 0;
                if (!new_axis && is_scalar)
                {
                    ORT_THROW("Cannot concatenate scalars");
                }

                const int32_t signedAxis = gsl::narrow_cast<int32_t>(kernelInfo.GetAttribute<int64_t>(AttrName::Axis));
                axis = static_cast<uint32_t>(HandleNegativeAxis(signedAxis, r + new_axis, !is_scalar));
                if (new_axis)
                {
                    ML_CHECK_VALID_ARGUMENT(axis < r + 1);
                    shape.insert(shape.begin() + axis, 1);
                }
                else
                {
                    ML_CHECK_VALID_ARGUMENT(axis < r);
                }

                // When processing the first tensor, initialize output shape, inputDimCount and axis.
                if (!inputDimCount)
                {
                    m_outputShape = shape;
                    inputDimCount = r + new_axis;
                }

                axisTotal += shape[axis];

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    continue;
                }

                ML_CHECK_BOOL(*inputDimCount == shape.size());
                m_inputTensorDescs.emplace_back(TensorDesc(sequenceInputDmlDataType, shape));
                m_inputIndices.push_back(i);
            }

            m_outputShape[axis] = axisTotal;

            // We should only call join if there exists input tensors that are non-empty and non-scalar.
            // In that case, the inputDimCount must be set and greater than 0.
            if (m_inputIndices.size() > 0)
            {
                m_outputTensorDesc = TensorDesc(sequenceInputDmlDataType, m_outputShape);
                auto dmlAxis = GetDmlAdjustedAxis(axis, *inputDimCount, m_outputTensorDesc.GetDimensionCount());

                auto outputIndices = std::vector<std::optional<uint32_t>> { 0 };
                gsl::span<const uint32_t> outputShapes[1] = { m_outputShape };
                DmlOperator::InitializeOutputsWithShapes(kernelInfo, outputIndices, outputShapes, 1);

                auto outputDescs = std::vector<DML_TENSOR_DESC> { m_outputTensorDesc.GetDmlDesc() };
                auto inputDescs = std::vector<DML_TENSOR_DESC>(m_inputTensorDescs.size());
                for (size_t i = 0; i < inputDescs.size(); i++)
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
        auto outputTensor = kernelContext.GetOutputTensor(0, m_outputShape).GetInterface().Get();

        if (!m_inputIndices.size())
        {
            return;
        }

        ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        kernelContext.GetInterface().As(&operatorKernelContext);
        auto inputTensors = std::vector<IMLOperatorTensor*>(m_inputIndices.size());
        for (uint32_t i = 0; i < inputTensors.size(); i++)
        {
            assert(m_inputTensorDescs[i].IsValid());
            ComPtr<IMLOperatorTensor> inputTensor;
            ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceInputTensor(0, m_inputIndices[i], &inputTensor));
            inputTensors[i] = inputTensor.Get();
        }

        auto outputTensors = gsl::span<IMLOperatorTensor*> { &outputTensor, 1 };

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            kernelContext.GetAllocator(),
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            outputTensors));
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ConcatFromSequence, DmlOperatorConcatFromSequence);

} // namespace Dml
