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

    void Shift1DInputsTensorDesc(
        const MLOperatorKernelCreationContext& kernelCreationContext,
        int index,
        int count,
        uint32_t minimumDimensionCount
        )
    {
        for (int i = index; i != index + count; ++i)
        {
            // Shift a single dimension size to the C channel.
            // e.g. Given a 4D input (X), then a 1D scale of [7] becomes [1,7,1,1].
            TensorDesc& tensorDesc = m_inputTensorDescs[i];
            gsl::span<const uint32_t> sizes = tensorDesc.GetSizes();
            gsl::span<const uint32_t> lastDimension = sizes.last(1);
            m_inputTensorDescs[i] = CreateTensorDescFromInput(kernelCreationContext, i, TensorAxis::DoNotCoerce, TensorAxis::C, TensorAxis::LeftAligned, lastDimension, minimumDimensionCount);
        }
    }

public:
    DmlOperatorInstanceNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(
            kernelCreationContext,
            kernelInputIndices,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            /*minimumDimensionCount*/ 1);

        const uint32_t dmlDimensionCount = std::max<uint32_t>(4u, m_inputTensorDescs[0].GetDimensionCount());

        // Shift IN_SCALE and IN_BIAS input tensor descs {1, C, 1, 1} out of 1D tensors.
        Shift1DInputsTensorDesc(kernelCreationContext, IN_SCALE, 2, dmlDimensionCount);

        // Pad the input and the output with trailing 1's until they are at least 4D
        auto sizes = m_inputTensorDescs[0].GetSizes();
        std::vector<uint32_t> tensorShape(sizes.begin(), sizes.end());
        tensorShape.resize(static_cast<size_t>(dmlDimensionCount), 1);
        m_inputTensorDescs[0] = TensorDesc(
            m_inputTensorDescs[0].GetDmlDataType(),
            tensorShape);
        m_outputTensorDescs[0] = m_inputTensorDescs[0];

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
