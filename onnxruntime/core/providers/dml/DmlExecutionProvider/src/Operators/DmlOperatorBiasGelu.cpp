// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorBiasGelu : public DmlOperator
{
public:
    DmlOperatorBiasGelu(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        // Broadcast bias to have the same dimensions as the input
        std::vector<uint32_t> inputTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, inputTensorShape);
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() == 2);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 1);

        TensorDesc biasInputTensorDesc(m_inputTensorDescs[0].GetDmlDataType(), m_inputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC biasInputDmlTensorDesc = biasInputTensorDesc.GetDmlDesc();

        DML_ACTIVATION_GELU_OPERATOR_DESC geluDesc = {};
        DML_OPERATOR_DESC geluOpDesc = { DML_OPERATOR_ACTIVATION_GELU, &geluDesc };

        DML_ELEMENT_WISE_ADD1_OPERATOR_DESC addDesc = {};
        addDesc.ATensor = &inputDescs[0];
        addDesc.BTensor = &inputDescs[1];
        addDesc.FusedActivation = &geluOpDesc;
        addDesc.OutputTensor = &outputDescs[0];
        DML_OPERATOR_DESC addOpDesc = { DML_OPERATOR_ELEMENT_WISE_ADD1, &addDesc };

        SetDmlOperatorDesc(addOpDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(BiasGelu, DmlOperatorBiasGelu);

} // namespace Dml
