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
        std::vector<DimensionType> inputShape0Broadcasted;
        std::vector<DimensionType> inputShape1Broadcasted;

        OperatorHelper::MatMulShapeMapping(inputShape0, inputShape1, outputShape, inputShape0Broadcasted, inputShape1Broadcasted);

        // Initialize the input descriptions without broadcasting yet, since MatMul has special rules where broadcasting the
        // original shape (notably when 1D) to the output shape would mess up because the dimensions are shifted.
        m_inputTensorDescs[0] = CreateTensorDescFromInput(kernelInfo, 0);
        m_inputTensorDescs[1] = CreateTensorDescFromInput(kernelInfo, 1);

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        // Broadcast the inputs to their broadcasted shapes.
        m_inputTensorDescs[0].SetBroadcastedShape(inputShape0Broadcasted, inputShape0, outputShape.size());
        m_inputTensorDescs[1].SetBroadcastedShape(inputShape1Broadcasted, inputShape1, outputShape.size());

        // DirectML only supports ranks up to 4D for GEMM, and so any leading dimensions must be folded.
        m_inputTensorDescs[0].SetDimensionCount(4, TensorAxis::RightAligned, /*foldEndDimensions*/ true);
        m_inputTensorDescs[1].SetDimensionCount(4, TensorAxis::RightAligned, /*foldEndDimensions*/ true);
        m_outputTensorDescs[0].SetDimensionCount(4, TensorAxis::RightAligned, /*foldEndDimensions*/ true);

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
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedMatMul, DmlOperatorMatMul);

} // namespace Dml
