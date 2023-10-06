// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorQLinearAdd : public DmlOperator
{
    enum InputTensors { 
        IN_A, 
        IN_A_SCALE,
        IN_A_ZERO_POINT, 
        IN_B, 
        IN_B_SCALE,
        IN_B_ZERO_POINT,
        IN_C_SCALE, 
        IN_C_ZERO_POINT 
    };

public:
    DmlOperatorQLinearAdd(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo)
    {
        DmlOperator::Initialize(kernelInfo);

        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        uint32_t dmlDimSize = m_inputTensorDescs[0].GetDimensionCount();

        // Initialize the input descriptions with broadcasting
        m_inputTensorDescs[IN_A] = CreateTensorDescFromInput(kernelInfo, 0/*A OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);
        m_inputTensorDescs[IN_B] = CreateTensorDescFromInput(kernelInfo, 3/*B OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        m_inputTensorDescs[IN_A_SCALE] = CreateTensorDescFromInput(kernelInfo, 1/*A Scale OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, dmlDimSize);
        m_inputTensorDescs[IN_A_ZERO_POINT] = CreateTensorDescFromInput(kernelInfo, 2/*A Zero point OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, dmlDimSize);

        m_inputTensorDescs[IN_B_SCALE] = CreateTensorDescFromInput(kernelInfo, 4/*B Scale OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, dmlDimSize);
        m_inputTensorDescs[IN_B_ZERO_POINT] = CreateTensorDescFromInput(kernelInfo, 5/*B Zero point OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, dmlDimSize);

        m_inputTensorDescs[IN_C_SCALE] = CreateTensorDescFromInput(kernelInfo, 6/*C Zero point OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, dmlDimSize);
        m_inputTensorDescs[IN_C_ZERO_POINT] = CreateTensorDescFromInput(kernelInfo, 7/*C Zero point OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, dmlDimSize);

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC AddDesc = {};
        AddDesc.ATensor = &inputDescs[IN_A];
        AddDesc.AScaleTensor = &inputDescs[IN_A_SCALE];
        AddDesc.AZeroPointTensor = inputDescs[IN_A_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_A_ZERO_POINT] : nullptr;
        AddDesc.BTensor = &inputDescs[IN_B];
        AddDesc.BScaleTensor = &inputDescs[IN_B_SCALE];
        AddDesc.BZeroPointTensor = inputDescs[IN_B_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_B_ZERO_POINT] : nullptr;
        AddDesc.OutputScaleTensor = &inputDescs[IN_C_SCALE];
        AddDesc.OutputZeroPointTensor = inputDescs[IN_C_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_C_ZERO_POINT] : nullptr; 
        AddDesc.OutputTensor = &outputDescs[0];
        
        TryConvertTensorToBroadcastScalar(kernelInfo, AddDesc.AScaleTensor,           IN_A_SCALE);
        TryConvertTensorToBroadcastScalar(kernelInfo, AddDesc.AZeroPointTensor,       IN_A_ZERO_POINT);

        TryConvertTensorToBroadcastScalar(kernelInfo, AddDesc.BScaleTensor,           IN_B_SCALE);
        TryConvertTensorToBroadcastScalar(kernelInfo, AddDesc.BZeroPointTensor,       IN_B_ZERO_POINT);

        TryConvertTensorToBroadcastScalar(kernelInfo, AddDesc.OutputScaleTensor,      IN_C_SCALE);
        TryConvertTensorToBroadcastScalar(kernelInfo, AddDesc.OutputZeroPointTensor,  IN_C_ZERO_POINT);

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD, &AddDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(QLinearAdd, DmlOperatorQLinearAdd);

} // namespace Dml
