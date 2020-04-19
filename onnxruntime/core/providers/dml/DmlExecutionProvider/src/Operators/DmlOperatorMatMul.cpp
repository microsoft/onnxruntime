// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorMatMul : public DmlOperator
{
    enum InputTensors { IN_A, IN_B };

public:
    DmlOperatorMatMul(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo)
    {
        // MatMul has two inputs, but DML GEMM requires 3 input bindings (a null binding for the C Tensor).
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 2);
        std::vector<std::optional<uint32_t>> inputIndices = { 0, 1, std::nullopt };
        DmlOperator::Initialize(kernelInfo, inputIndices);

        std::vector<DimensionType> inputShape0 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<DimensionType> inputShape1 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(1);
        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        OperatorHelper::MatMulShapeMapping(inputShape0, inputShape1, outputShape);

        // Initialize the input descriptions with broadcasting
        m_inputTensorDescs[0] = CreateTensorDescFromInput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape0);
        m_inputTensorDescs[1] = CreateTensorDescFromInput(kernelInfo, 1, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape1);

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        DML_GEMM_OPERATOR_DESC gemmDesc = {};
        gemmDesc.ATensor = &inputDescs[0];
        gemmDesc.BTensor = &inputDescs[1];
        gemmDesc.CTensor = nullptr;
        gemmDesc.OutputTensor = &outputDescs[0];
        gemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.Alpha = 1.0f;
        gemmDesc.Beta = 0.0f;
        gemmDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_GEMM, &gemmDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};


DML_OP_DEFINE_CREATION_FUNCTION(MatMul, DmlOperatorMatMul);
DML_OP_DEFINE_CREATION_FUNCTION(FusedMatMul, DmlOperatorMatMul);

} // namespace Dml
