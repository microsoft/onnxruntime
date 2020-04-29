// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorMaxUnpool : public DmlOperator, public UnpoolingHelper
{
public:
    using Self = DmlOperatorMaxUnpool;

    DmlOperatorMaxUnpool(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        UnpoolingHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        uint32_t inputCount = kernelCreationContext.GetInputCount();
        ML_CHECK_VALID_ARGUMENT(inputCount == 2 || inputCount == 3, "MaxUnpool expects 2 or 3 inputs.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "MaxUnpool expects 1 output.");

        std::vector<std::optional<uint32_t>> inputIndices = { 0, 1 }; // The 3rd tensor ('output_shape') is not bound, just 'X' and 'I' indices.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        assert(inputDescs.size() == 2);
        assert(outputDescs.size() == 1);

        DML_MAX_UNPOOLING_OPERATOR_DESC poolingDesc = {};
        poolingDesc.InputTensor = &inputDescs[0];
        poolingDesc.IndicesTensor = &inputDescs[1];
        poolingDesc.OutputTensor = outputDescs.data();

        DML_OPERATOR_DESC operaterDesc = {};
        operaterDesc.Type = DML_OPERATOR_MAX_UNPOOLING;
        operaterDesc.Desc = &poolingDesc;
        SetDmlOperatorDesc(operaterDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MaxUnpool, DmlOperatorMaxUnpool);

} // namespace Dml
