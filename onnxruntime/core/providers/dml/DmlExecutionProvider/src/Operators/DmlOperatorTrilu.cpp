// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "./precomp.h"

namespace Dml
{

class DmlOperatorTrilu : public DmlOperator
{
public:
    explicit DmlOperatorTrilu(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 1, "Trilu expects 1-2 inputs.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Trilu expects 1 output.");

        std::vector<std::optional<uint32_t>> inputIndices = {0};  // Use only the first tensor. The second tensor is CPU-based (k).
        std::vector<std::optional<uint32_t>> outputIndices = {0};
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        assert(inputDescs.size() == 1);
        assert(outputDescs.size() == 1);

        // Read the diagonal offset from the 2nd tensor (defaults to 0 if absent).
        int32_t k = 0;
        if (kernelCreationContext.IsInputValid(1))
        {
            MLOperatorTensor kTensor = kernelCreationContext.GetConstantInputTensor(1);
            k = gsl::narrow_cast<int32_t>(ReadScalarTensorCastToInt64(kTensor));
        }

        auto outputTensorShapeDescription = kernelCreationContext.GetTensorShapeDescription();
        std::vector<DimensionType> outputDimensions = outputTensorShapeDescription.GetOutputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(outputDimensions.size() <= OperatorHelper::NchwDimensionCount);

        const bool keepUpperDiagonal = kernelCreationContext.GetOptionalAttribute<bool>(AttrName::Upper, 0);

        DML_DIAGONAL_MATRIX1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.DiagonalFillBegin = keepUpperDiagonal ? INT32_MIN : k + 1;
        operatorDesc.DiagonalFillEnd = keepUpperDiagonal ? k : INT32_MAX;
        operatorDesc.ValueDataType = m_inputTensorDescs[0].GetDmlDataType();
        CastToClampedScalarUnion<float>(operatorDesc.ValueDataType, 0.0f, /*out*/&operatorDesc.Value);

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_DIAGONAL_MATRIX1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Trilu, DmlOperatorTrilu);

}  // namespace Dml
