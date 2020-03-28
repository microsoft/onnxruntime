// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorQLinearMatMul : public DmlOperator
{
public:
    using Self = DmlOperatorQLinearMatMul;

    DmlOperatorQLinearMatMul(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
#if 0 // TODO:NickFe - https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearmatmul
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 8);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_PLACEHOLDER_OPERATOR_DESC operatorDesc = {};
        operatorDesc.IndicesTensor = &inputDescs[0];
        operatorDesc.ValuesTensor = &inputDescs[1];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axis = dmlAxis;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_PLACEHOLDER, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
#endif
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(QLinearMatMul, DmlOperatorQLinearMatMul);

} // namespace Dml
