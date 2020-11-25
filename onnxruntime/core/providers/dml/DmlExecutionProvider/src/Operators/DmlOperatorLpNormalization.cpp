// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorLpNormalization : public DmlOperator
{
public:
    DmlOperatorLpNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        DmlOperator::Initialize(kernelCreationContext);

        const int onnxAxis = kernelCreationContext.GetOptionalAttribute<int>(AttrName::Axis, 0);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // Valid values for p are 1 and 2.
        int p = kernelCreationContext.GetOptionalAttribute<int>(AttrName::P, 2);
        ML_CHECK_VALID_ARGUMENT(p >= 1 && p <= 2);

        uint32_t dmlAxis = GetDmlAdjustedAxis(onnxAxis, kernelCreationContext, m_inputTensorDescs.front().GetDimensionCount());

        DML_LP_NORMALIZATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axis = dmlAxis;
        operatorDesc.Epsilon = DefaultEpsilon;
        operatorDesc.P = p;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_LP_NORMALIZATION, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(LpNormalization, DmlOperatorLpNormalization);

} // namespace Dml
