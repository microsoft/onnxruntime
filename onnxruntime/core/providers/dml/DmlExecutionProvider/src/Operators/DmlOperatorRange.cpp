// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "precomp.h"

namespace Dml
{

class DmlOperatorRange : public DmlOperator, RangeHelper
{
public:
    using Self = DmlOperatorRange;

    DmlOperatorRange(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        RangeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        std::vector<std::optional<uint32_t>> inputIndices = {}; // All tensors are CPU bound.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC operatorDesc = {};
        operatorDesc.ValueDataType = m_outputTensorDescs[0].GetDmlDataType();;
        static_assert(sizeof(operatorDesc.ValueStart) == sizeof(m_valueStart));
        static_assert(sizeof(operatorDesc.ValueDelta) == sizeof(m_valueDelta));
        memcpy(&operatorDesc.ValueStart, &m_valueStart, sizeof(m_valueStart));
        memcpy(&operatorDesc.ValueDelta, &m_valueDelta, sizeof(m_valueDelta));
        operatorDesc.OutputTensor = outputDescs.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Range, DmlOperatorRange);

} // namespace Dml
