// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorInstanceNormalization : public DmlOperator
{
    enum InputTensors
    {
        IN_X,
        IN_SCALE,
        IN_BIAS
    };

public:
    DmlOperatorInstanceNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ORT_THROW_HR_IF(E_INVALIDARG, kernelCreationContext.GetInputCount() != 3);
        std::vector<std::vector<uint32_t>> inputShapes(kernelCreationContext.GetInputCount());
        std::vector<gsl::span<const uint32_t>> inputShapeSpans(kernelCreationContext.GetInputCount());
        auto firstInputShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);

        DmlOperator::Initialize(
            kernelCreationContext,
            std::nullopt,
            std::nullopt,
            firstInputShape,
            firstInputShape,
            static_cast<uint32_t>(firstInputShape.size()));

        // To allow metacommands to be called, add missing trailing dimensions of 1 until we reach 4 dimensions
        firstInputShape.resize(std::max<size_t>(firstInputShape.size(), 4), 1);

        m_inputTensorDescs[0] = TensorDesc(
            m_inputTensorDescs[0].GetDmlDataType(),
            firstInputShape);

        m_outputTensorDescs[0] = m_inputTensorDescs[0];

        for (uint32_t i = 1; i < m_inputTensorDescs.size(); ++i)
        {
            // Only the channel dimension shouldn't be broadcasted
            std::vector<uint32_t> inputStrides(firstInputShape.size());
            inputStrides[1] = 1;

            m_inputTensorDescs[i] = TensorDesc(
                m_inputTensorDescs[i].GetDmlDataType(),
                firstInputShape,
                inputStrides);
        }

        // "Instance" normalization is really spatial normalization,
        // where the spatial channels are reduced and normalized, while
        // batch and channel remain independent. So pass a list of axes
        // just beyond the leading batch and channel dimensions (starting
        // at axis 2 up to the last spatial dimension).
        const uint32_t inputDimensionCount = m_inputTensorDescs.front().GetDimensionCount();
        std::vector<uint32_t> axes(inputDimensionCount - 2);
        std::iota(axes.begin(), axes.end(), 2u);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        const std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = &inputDescs[2];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axes = axes.data();
        operatorDesc.AxisCount = static_cast<uint32_t>(axes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(InstanceNormalization, DmlOperatorInstanceNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedInstanceNormalization, DmlOperatorInstanceNormalization);

} // namespace Dml
