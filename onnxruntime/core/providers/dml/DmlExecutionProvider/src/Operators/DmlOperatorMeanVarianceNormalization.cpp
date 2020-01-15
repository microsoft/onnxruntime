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
        const bool normalizeVariance = (static_cast<bool>(kernelCreationContext.GetOptionalAttribute<int>(AttrName::NormalizeVariance, 0)));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = nullptr;
        operatorDesc.BiasTensor = nullptr;
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.CrossChannel = acrossChannels;
        operatorDesc.NormalizeVariance = normalizeVariance;
        operatorDesc.Epsilon = DefaultEpsilon;
        operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MeanVarianceNormalization, DmlOperatorMeanVarNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(FusedMeanVarianceNormalization, DmlOperatorMeanVarNormalization);

} // namespace Dml
