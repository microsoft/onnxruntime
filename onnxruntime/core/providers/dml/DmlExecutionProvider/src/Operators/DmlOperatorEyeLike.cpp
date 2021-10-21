// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorEyeLike : public DmlOperator
{
public:
    DmlOperatorEyeLike(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "EyeLike expects 1 input.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "EyeLike expects 1 output.");

        std::vector<std::optional<uint32_t>> inputIndices = {}; // Ignore the 1st input tensor for the GPU.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        assert(inputDescs.size() <= 1);
        assert(outputDescs.size() == 1);

        auto outputTensorShapeDescription = kernelCreationContext.GetTensorShapeDescription();;
        std::vector<DimensionType> outputDimensions = outputTensorShapeDescription.GetOutputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(outputDimensions.size() <= OperatorHelper::NchwDimensionCount);

        const int32_t diagonalOffset = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::K, 0);

        DML_DIAGONAL_MATRIX_OPERATOR_DESC operatorDesc = {};
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Offset = diagonalOffset;
        operatorDesc.Value = 1.0f;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_DIAGONAL_MATRIX, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(EyeLike, DmlOperatorEyeLike);

} // namespace Dml
