// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorCumSum : public DmlOperator, OneHotHelper
{
public:
    using Self = DmlOperatorCumSum;

    DmlOperatorCumSum(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        OneHotHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelCreationContext);
        
        // Adjust the axis so it's in DML's terms rather than the original ONNX indexing.
        int32_t hasExclusiveSum = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Exclusive, 0);
        int32_t isReversed = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Reverse, 0);
        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        uint32_t dmlAxis = GetDmlAdjustedAxis(onnxAxis, kernelCreationContext, m_inputTensorDescs.front().GetDimensionCount());
        
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_CUMULATIVE_SUMMATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.HasExclusiveSum = hasExclusiveSum;
        operatorDesc.Axis = dmlAxis;
        operatorDesc.AxisDirection = isReversed ? DML_AXIS_DIRECTION_DECREASING : DML_AXIS_DIRECTION_INCREASING;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CUMULATIVE_SUMMATION, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(CumSum, DmlOperatorCumSum);

} // namespace Dml
