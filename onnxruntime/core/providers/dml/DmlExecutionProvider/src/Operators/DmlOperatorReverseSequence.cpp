// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "precomp.h"

namespace Dml
{

class DmlOperatorReverseSequence : public DmlOperator
{
public:
    using Self = DmlOperatorReverseSequence;

    DmlOperatorReverseSequence(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelCreationContext);

        std::vector<uint32_t> inputDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<uint32_t> sequenceLengthDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(1);
        const uint32_t inputRank = static_cast<uint32_t>(inputDimensions.size());

        // Read axis.
        const int32_t batchAxis = HandleNegativeAxis(kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::BatchAxis, 0), inputRank);
        const int32_t timeAxis = HandleNegativeAxis(kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::TimeAxis, 0), inputRank);
        const uint32_t dmlTimeAxis = GetDmlAdjustedAxis(timeAxis, inputRank, m_inputTensorDescs.front().GetDimensionCount());
        ML_CHECK_VALID_ARGUMENT(timeAxis != batchAxis);

        // Fix up the sequence lengths tensor (originally 1D) to be rank compatible with input,
        // with all dimensions being the same as input except the active reversal axis.
        std::vector<uint32_t> adjustedSequenceLengthDimensions = inputDimensions;
        adjustedSequenceLengthDimensions[timeAxis] = 1;
        ML_CHECK_VALID_ARGUMENT(ComputeElementCountFromDimensions(adjustedSequenceLengthDimensions), ComputeElementCountFromDimensions(sequenceLengthDimensions));

        m_inputTensorDescs[1] =
            TensorDesc(
                m_inputTensorDescs[1].GetMlOperatorDataType(),
                gsl::make_span(adjustedSequenceLengthDimensions),
                gsl::make_span(adjustedSequenceLengthDimensions),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0
            );

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.SequenceLengthsTensor = &inputDescs[1];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axis = dmlTimeAxis;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_REVERSE_SUBSEQUENCES, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ReverseSequence, DmlOperatorReverseSequence);

} // namespace Dml
