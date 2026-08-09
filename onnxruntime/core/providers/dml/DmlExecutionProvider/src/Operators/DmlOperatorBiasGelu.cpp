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
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1 || kernelCreationContext.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        // Broadcast bias to have the same dimensions as the input
        std::vector<uint32_t> inputTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, inputTensorShape);
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() == kernelCreationContext.GetInputCount());
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 1);

        if (kernelCreationContext.IsInputValid(1))
        {
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
        else
        {
            DML_ACTIVATION_GELU_OPERATOR_DESC geluDesc;
            geluDesc.InputTensor = &inputDescs[0];
            geluDesc.OutputTensor = &outputDescs[0];
            DML_OPERATOR_DESC geluOpDesc = { DML_OPERATOR_ACTIVATION_GELU, &geluDesc };
            SetDmlOperatorDesc(geluOpDesc, kernelCreationContext);
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(BiasGelu, DmlOperatorBiasGelu);
DML_OP_DEFINE_CREATION_FUNCTION(FastGelu, DmlOperatorBiasGelu);

} // namespace Dml
