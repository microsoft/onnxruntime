// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorDynamicQuantizeLinear : public DmlOperator
{
public:
    using Self = DmlOperatorDynamicQuantizeLinear;

    DmlOperatorDynamicQuantizeLinear(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
#if 0 // TODO:NickFe - https://github.com/onnx/onnx/blob/master/docs/Operators.md#dynamicquantizelinear
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 3);
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

DML_OP_DEFINE_CREATION_FUNCTION(DynamicQuantizeLinear, DmlOperatorDynamicQuantizeLinear);

} // namespace Dml
