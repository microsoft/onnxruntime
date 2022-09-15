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

        // Initialize Input, Scale and Bias tensors with same dimension count as Input tensor
        // because DML MVN1 has a validation which requires all 3 needs to have same dimension count
        // due to historical artifact.
        DmlOperator::Initialize(
            kernelCreationContext, 
            kernelInputIndices,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0));

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);

        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        uint32_t inputDimCount = kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
        onnxAxis = OperatorHelper::HandleNegativeAxis(onnxAxis, inputDimCount);
        std::vector<uint32_t> onnxAxes(inputDimCount - onnxAxis);
        std::iota(onnxAxes.begin(), onnxAxes.end(), onnxAxis);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = inputDescs[2].Desc != nullptr ? &inputDescs[2] : nullptr;
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axes = onnxAxes.data();
        operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(onnxAxes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

void CALLBACK QueryLayerNormalization(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;

    // Mean and InvStdDev are not supported outputs.
    // If only Scale tensor is present then fall back to CPU. This is temporary until 
    // DML1.9.2 or DML1.10 gets released.
    if (context->GetInputCount() < 3 || context->GetOutputCount() > 1) 
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(LayerNormalization, DmlOperatorLayerNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(LayerNormalization17, DmlOperatorLayerNormalization);

} // namespace Dml
