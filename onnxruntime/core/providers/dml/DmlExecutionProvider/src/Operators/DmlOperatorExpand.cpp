// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorExpand : public DmlOperator, public ExpandHelper
{
public:
    using Self = DmlOperatorExpand;

    DmlOperatorExpand(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        ExpandHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> inputIndices = { 0 }; // The second tensor is not bound to Identity operator.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        Initialize(kernelCreationContext, inputIndices, outputIndices);

        TensorDesc inputTensorDesc = 
            TensorDesc(
                kernelCreationContext.GetInputEdgeDescription(0).tensorDataType,
                m_outputTensorDescs[0].GetSizes(),
                m_inputTensorDescs[0].GetSizes(),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0);
        
        TensorDesc outputTensorDesc = 
            TensorDesc(
                kernelCreationContext.GetOutputEdgeDescription(0).tensorDataType,
                m_outputTensorDescs[0].GetSizes(),
                m_outputTensorDescs[0].GetSizes(),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0
            );

        m_inputTensorDescs[0]  = inputTensorDesc;
        m_outputTensorDescs[0] = outputTensorDesc;

        // Create the operator with new shape after calling UpdateShape.
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.OutputTensor = &outputDescs[0];
        // identityDesc.ScaleBias left empty.

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &operatorDesc}, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Expand, DmlOperatorExpand);

} // namespace Dml
