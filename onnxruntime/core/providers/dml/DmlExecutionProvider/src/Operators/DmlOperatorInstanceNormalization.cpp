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

    void Shift1DInputsTensorDesc(const MLOperatorKernelCreationContext& kernelCreationContext, int index, int count, uint32_t destinationAxis)
    {
        for (int i = index; i != index + count; ++i)
        {
            // Shift a single dimension size to the C channel.
            // e.g. [7] or [1,1,1,7] becomes [1,7,1,1]
            TensorDesc& tensorDesc = m_inputTensorDescs[i];
            gsl::span<const uint32_t> sizes = tensorDesc.GetSizes();
            gsl::span<const uint32_t> lastDimension = sizes.last(1);
            ML_CHECK_VALID_ARGUMENT(tensorDesc.GetDimensionCount() == OperatorHelper::NchwDimensionCount);
            ML_CHECK_VALID_ARGUMENT(sizes.size() >=4 && sizes[N] == 1 && sizes[C] == 1 && sizes[H] == 1);
            m_inputTensorDescs[i] = CreateTensorDescFromInput(kernelCreationContext, i, TensorAxis::DoNotCoerce, TensorAxis::C, TensorAxis::LeftAligned, lastDimension);
        }
    }

public:
    DmlOperatorInstanceNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        const std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        // Shift IN_SCALE and IN_BIAS input tensor descs {1, C, 1, 1} out of 1D tensors.
        Shift1DInputsTensorDesc(kernelCreationContext, IN_SCALE, 2, C);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = &inputDescs[2];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.CrossChannel = false;
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(InstanceNormalization, DmlOperatorInstanceNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(FusedInstanceNormalization, DmlOperatorInstanceNormalization);

} // namespace Dml
