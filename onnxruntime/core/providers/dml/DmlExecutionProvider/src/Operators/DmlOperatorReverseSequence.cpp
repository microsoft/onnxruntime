// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


// TODO:::

#include "precomp.h"

namespace Dml
{

class DmlOperatorReverseSequence : public DmlOperator, OneHotHelper
{
public:
    using Self = DmlOperatorReverseSequence;

    DmlOperatorReverseSequence(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        OneHotHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelCreationContext);

        std::vector<uint32_t> inputDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<uint32_t> sequenceLengthDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(1);

        // Read axis.
        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::TimeAxis, 0);
        onnxAxis = HandleNegativeAxis(onnxAxis, static_cast<uint32_t>(inputDimensions.size()));
        const uint32_t dmlAxis = GetDmlAdjustedAxis(onnxAxis, onnxAxis, m_inputTensorDescs.front().GetDimensionCount());

        // Fix up the sequence lengths tensor (originally 1D) to be rank compatible with input,
        // with all dimensions being the same as input except the active reversal axis.
        std::vector<uint32_t> adjustedSequenceLengthDimensions = inputDimensions;
        adjustedSequenceLengthDimensions[onnxAxis] = 1;
        ML_CHECK_VALID_ARGUMENT(ComputeElementCountFromDimensions(adjustedSequenceLengthDimensions), ComputeElementCountFromDimensions(sequenceLengthDimensions));

        m_inputTensorDescs[1] =
            TensorDesc(
                m_inputTensorDescs[0].GetMlOperatorDataType(),
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
        operatorDesc.Axis = dmlAxis;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_REVERSE_SUBSEQUENCES, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ReverseSequence, DmlOperatorReverseSequence);

} // namespace Dml
