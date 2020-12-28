// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorBatchNormalization : public DmlOperator
{
    // This order matches the ONNX schema.
    enum OnnxInputIndex
    {
        X, // Input
        Scale,
        Bias,
        Mean,
        Variance,
        Count,
    };

public:
    DmlOperatorBatchNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {X, Mean, Variance, Scale, Bias};
        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 5);
        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs.size() >= 1);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, 0.0f);
        const int spatial = kernelCreationContext.GetOptionalAttribute<int>(AttrName::Spatial, 1);
        const std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        m_inputTensorDescs[0] = CreateTensorDescFromInput(kernelCreationContext, 0, TensorAxis::DoNotCoerce, TensorAxis::N, TensorAxis::LeftAligned);

        // Massage each of these 1D tensors (of length C) into ND tensors of the form [1,C,1,1,...].
        for (uint32_t i = Scale; i < OnnxInputIndex::Count; ++i)
        {
            m_inputTensorDescs[i] = CreateTensorDescFromInput(kernelCreationContext, i, TensorAxis::DoNotCoerce, TensorAxis::C, TensorAxis::LeftAligned, std::nullopt, m_inputTensorDescs[0].GetDimensionCount());
        }

        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelCreationContext, 0, TensorAxis::DoNotCoerce, TensorAxis::N, TensorAxis::LeftAligned, std::nullopt, m_inputTensorDescs[0].GetDimensionCount());

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_BATCH_NORMALIZATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[X];
        operatorDesc.MeanTensor = &inputDescs[Mean];
        operatorDesc.VarianceTensor = &inputDescs[Variance];
        operatorDesc.ScaleTensor = &inputDescs[Scale];
        operatorDesc.BiasTensor = &inputDescs[Bias];
        operatorDesc.OutputTensor = &outputDescs[0];
        operatorDesc.Spatial = static_cast<BOOL>(spatial);
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_BATCH_NORMALIZATION, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(BatchNormalization, DmlOperatorBatchNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(FusedBatchNormalization, DmlOperatorBatchNormalization);

} // namespace Dml
