// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorMeanVarNormalization : public DmlOperator
{
public:
    DmlOperatorMeanVarNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, std::nullopt, std::nullopt};
        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

        const bool acrossChannels = (static_cast<bool>(kernelCreationContext.GetOptionalAttribute<int>(AttrName::AcrossChannels, 0)));
        const bool normalizeVariance = (static_cast<bool>(kernelCreationContext.GetOptionalAttribute<int>(AttrName::NormalizeVariance, 1)));

        // If not specified, the default axes are [0,2,3].
        std::vector<int32_t> onnxAxes = kernelCreationContext.GetOptionalAttributeVectorInt32(AttrName::Axes);
        if (onnxAxes.empty())
        {
            int32_t crossChannelAxes[] = { 0, 1, 2, 3 };
            int32_t nonChannelAxes[] = {0, 2, 3};
            gsl::span<int32_t> defaultAxes(acrossChannels ? gsl::make_span(crossChannelAxes) : gsl::make_span(nonChannelAxes));
            onnxAxes.assign(defaultAxes.begin(), defaultAxes.end());
        }

        std::vector<uint32_t> dmlAxes;
        const std::vector<DimensionType> inputDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        const int32_t inputDimCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        GetDmlAdjustedAxes(onnxAxes, inputDimCount, m_inputTensorDescs.front().GetDimensionCount(), /*out*/ dmlAxes);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = nullptr;
        operatorDesc.BiasTensor = nullptr;
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axes = dmlAxes.data();
        operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());
        operatorDesc.NormalizeVariance = normalizeVariance;
        operatorDesc.Epsilon = DefaultEpsilon;
        operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MeanVarianceNormalization, DmlOperatorMeanVarNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(FusedMeanVarianceNormalization, DmlOperatorMeanVarNormalization);

} // namespace Dml
