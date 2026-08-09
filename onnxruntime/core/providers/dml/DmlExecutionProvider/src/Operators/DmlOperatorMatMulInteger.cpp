// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorMatMulInteger : public DmlOperator
{
    enum InputTensors { 
        IN_A, 
        IN_A_ZERO_POINT, 
        IN_B, 
        IN_B_ZERO_POINT 
    };

public:
    DmlOperatorMatMulInteger(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo)
    {
        std::vector<std::optional<uint32_t>> inputIndices = { 0, 2, 1, 3 };
        DmlOperator::Initialize(kernelInfo, inputIndices);

        std::vector<DimensionType> inputShape0 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<DimensionType> inputShape1 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(1);
        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        OperatorHelper::MatMulShapeMapping(inputShape0, inputShape1, outputShape);

        // Initialize the input descriptions with broadcasting
        m_inputTensorDescs[IN_A] = CreateTensorDescFromInput(kernelInfo, 0/*OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape0);
        m_inputTensorDescs[IN_B] = CreateTensorDescFromInput(kernelInfo, 1/*OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape1);

        uint32_t dmlDimSize = m_inputTensorDescs[0].GetDimensionCount();
        // Resize the A ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        m_inputTensorDescs[IN_A_ZERO_POINT] = CreateTensorDescFromInput(
            kernelInfo, 
            2/*Onnx Index*/, 
            TensorAxis::DoNotCoerce, 
            TensorAxis::H,
            TensorAxis::LeftAligned,
            std::nullopt,
            dmlDimSize
            );
        
        // B Zeropoint and BScale are already aligned in the W dimension so no need to align them

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC matmulDesc = {};
        matmulDesc.ATensor = &inputDescs[IN_A];
        matmulDesc.AZeroPointTensor = inputDescs[IN_A_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_A_ZERO_POINT] : nullptr;
        matmulDesc.BTensor = &inputDescs[IN_B];
        matmulDesc.BZeroPointTensor = inputDescs[IN_B_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_B_ZERO_POINT] : nullptr;
        matmulDesc.OutputTensor = &outputDescs[0];

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MATRIX_MULTIPLY_INTEGER, &matmulDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};


DML_OP_DEFINE_CREATION_FUNCTION(MatMulInteger, DmlOperatorMatMulInteger);

} // namespace Dml
