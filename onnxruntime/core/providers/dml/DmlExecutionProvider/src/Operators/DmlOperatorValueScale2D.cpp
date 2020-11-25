// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

// https://github.com/onnx/onnx/blob/master/docs/Operators.md#experimental-imagescaler
class DmlOperatorValueScale2d : public DmlOperator
{
public:
    DmlOperatorValueScale2d(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        DmlOperator::Initialize(kernelCreationContext);

        std::vector<float> bias = kernelCreationContext.GetAttributeVector<float>(AttrName::Bias);
        float scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, 1.0f);

        std::vector<DimensionType> inputDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() == NchwDimensionCount, "Wrong number of dimensions.");
        ML_CHECK_VALID_ARGUMENT(inputDimensions[C] == bias.size(), "input dimension count for channel C must equal bias count.");

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() >= 1);

        DML_VALUE_SCALE_2D_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Scale = scale;
        operatorDesc.ChannelCount = gsl::narrow_cast<uint32_t>(bias.size());
        operatorDesc.Bias = bias.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_VALUE_SCALE_2D, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ImageScaler, DmlOperatorValueScale2d);

} // namespace Dml
