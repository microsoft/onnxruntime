// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorLayerNormalization : public DmlOperator
{
public:
    DmlOperatorLayerNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);

        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        uint32_t inputDimCount = kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
        onnxAxis = OperatorHelper::HandleNegativeAxis(onnxAxis, inputDimCount);
        std::vector<int32_t> onnxAxes(inputDimCount - onnxAxis);
        std::iota(onnxAxes.begin(), onnxAxes.end(), onnxAxis);

        std::vector<uint32_t> dmlAxes;
        GetDmlAdjustedAxes(onnxAxes, inputDimCount, m_inputTensorDescs.front().GetDimensionCount(), /*out*/ dmlAxes);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = inputDescs[2].Desc != nullptr ? &inputDescs[2] : nullptr;
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axes = dmlAxes.data();
        operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

void CALLBACK QueryLayerNormalization(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = true;

    if (context->GetOutputCount() > 1)
    {
        *isSupported = false;
        return;
    }
}

DML_OP_DEFINE_CREATION_FUNCTION(LayerNormalization, DmlOperatorLayerNormalization);

} // namespace Dml
