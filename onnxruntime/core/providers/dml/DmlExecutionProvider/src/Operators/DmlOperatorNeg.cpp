// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorNeg : public DmlOperator
{
public:
    using Self = DmlOperatorNeg;

    DmlOperatorNeg(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_NEGATE, &opDesc}, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Neg, DmlOperatorNeg);

} // namespace Dml
