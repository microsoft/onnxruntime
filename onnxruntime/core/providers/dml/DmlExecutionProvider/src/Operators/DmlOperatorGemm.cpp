// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorGemm : public DmlOperator, public GemmHelper
{
public:
    DmlOperatorGemm(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo), 
            GemmHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 2);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelInfo);

        bool containsBiasTensor = kernelInfo.IsInputValid(2);

        // Broadcast C tensor to the shape of the output tensor.
        if (containsBiasTensor)
        {
            m_inputTensorDescs[2] = CreateTensorDescFromInput(
                kernelInfo,
                2,
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0)
            );
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        DML_GEMM_OPERATOR_DESC gemmDesc = {};
        gemmDesc.ATensor = &inputDescs[0];
        gemmDesc.BTensor = &inputDescs[1];
        gemmDesc.CTensor = kernelInfo.IsInputValid(2) ? &inputDescs[2] : nullptr;
        gemmDesc.OutputTensor = &outputDescs[0];
        gemmDesc.TransA = (m_transA ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE);
        gemmDesc.TransB = (m_transB ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE);
        gemmDesc.Alpha = m_alpha;
        gemmDesc.Beta = m_beta;
        gemmDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_GEMM, &gemmDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Gemm, DmlOperatorGemm);
DML_OP_DEFINE_CREATION_FUNCTION(FusedGemm, DmlOperatorGemm);

} // namespace Dml
