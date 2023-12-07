// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "./precomp.h"

namespace Dml
{

class DmlOperatorCol2Im : public DmlOperator, public Col2ImHelper
{
public:
    explicit DmlOperatorCol2Im(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        Col2ImHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3, "Col2Im expects 3 inputs.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Col2Im expects 1 output.");

        auto tensorShapeDescription = kernelCreationContext.GetTensorShapeDescription();
        std::vector<uint32_t> inputTensorShape = tensorShapeDescription.GetInputTensorShape(0);
        std::vector<uint32_t> outputTensorShape = tensorShapeDescription.GetOutputTensorShape(0);

        ML_CHECK_VALID_ARGUMENT(outputTensorShape == m_outputShape);

        std::vector<std::optional<uint32_t>> inputIndices = { 0 };
        gsl::span<const uint32_t> inputShapes[1] = { m_inputShape };
        gsl::span<const uint32_t> outputShapes[1] = { m_outputShape };
        DmlOperator::InitializeWithShapes(
            kernelCreationContext,
            inputIndices,
            std::nullopt,
            inputShapes,
            outputShapes,
            3
        );
        // Prepare DML_FOLD_OPERATOR_DESC
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        assert(inputDescs.size() == 1);
        assert(outputDescs.size() == 1);

        DML_FOLD_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(m_blockShape.size());
        operatorDesc.WindowSizes = m_blockShape.data();
        operatorDesc.Dilations = m_dilations.data();
        operatorDesc.StartPadding = m_pads.data();
        operatorDesc.EndPadding = m_pads.data();
        operatorDesc.Strides = m_strides.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_FOLD, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Col2Im, DmlOperatorCol2Im);

}  // namespace Dml
