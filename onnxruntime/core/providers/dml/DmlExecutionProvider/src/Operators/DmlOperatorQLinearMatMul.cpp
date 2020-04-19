// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorQLinearMatMul : public DmlOperator
{
    enum InputTensors { 
        IN_A, 
        IN_A_SCALE,
        IN_A_ZERO_POINT, 
        IN_B, 
        IN_B_SCALE,
        IN_B_ZERO_POINT,
        IN_Y_SCALE, 
        IN_Y_ZERO_POINT 
    };

public:
    DmlOperatorQLinearMatMul(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo)
    {
        std::vector<std::optional<uint32_t>> inputIndices = { 0, 1, 2, 3, 4, 5, 6, 7 };
        DmlOperator::Initialize(kernelInfo, inputIndices);

        std::vector<DimensionType> inputShape0 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0/*A OnnxIndex*/);
        std::vector<DimensionType> inputShape1 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(3/*B OnnxIndex*/);
        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        OperatorHelper::MatMulShapeMapping(inputShape0, inputShape1, outputShape);

        // Initialize the input descriptions with broadcasting
        m_inputTensorDescs[IN_A] = CreateTensorDescFromInput(kernelInfo, 0/*A OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape0);
        m_inputTensorDescs[IN_B] = CreateTensorDescFromInput(kernelInfo, 3/*B OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape1);

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC matMulDesc = {};
        matMulDesc.ATensor = &inputDescs[IN_A];
        matMulDesc.AScaleTensor = &inputDescs[IN_A_SCALE];
        matMulDesc.AZeroPointTensor = inputDescs[IN_A_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_A_ZERO_POINT] : nullptr;
        matMulDesc.BTensor = &inputDescs[IN_B];
        matMulDesc.BScaleTensor = &inputDescs[IN_B_SCALE];
        matMulDesc.BZeroPointTensor = inputDescs[IN_B_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_B_ZERO_POINT] : nullptr;
        matMulDesc.OutputScaleTensor = &inputDescs[IN_Y_SCALE];
        matMulDesc.OutputZeroPointTensor = inputDescs[IN_Y_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_Y_ZERO_POINT] : nullptr; 
        matMulDesc.OutputTensor = &outputDescs[0];

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY, &matMulDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};


DML_OP_DEFINE_CREATION_FUNCTION(QLinearMatMul, DmlOperatorQLinearMatMul);

} // namespace Dml
