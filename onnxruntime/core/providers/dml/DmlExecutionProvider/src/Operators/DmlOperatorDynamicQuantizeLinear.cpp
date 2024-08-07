// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorDynamicQuantizeLinear : public DmlOperator
{    
    enum DmlInputTensors { 
        IN_A, 
    };

    enum DmlOutputTensors { 
        OUT_Y,
        OUT_Y_SCALE, 
        OUT_Y_ZERO_POINT 
    };

public:
    using Self = DmlOperatorDynamicQuantizeLinear;

    DmlOperatorDynamicQuantizeLinear(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 3);

        DmlOperator::Initialize(kernelCreationContext);

        m_inputTensorDescs[IN_A] = CreateTensorDescFromInput(kernelCreationContext, 0/*A OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, NchwDimensionCount);

        m_outputTensorDescs[OUT_Y] = CreateTensorDescFromOutput(kernelCreationContext, 0/*Y OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, NchwDimensionCount);
        m_outputTensorDescs[OUT_Y_SCALE] = CreateTensorDescFromOutput(kernelCreationContext, 1/*Y Scale OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, NchwDimensionCount);
        m_outputTensorDescs[OUT_Y_ZERO_POINT] = CreateTensorDescFromOutput(kernelCreationContext, 2/*Y Zero point OnnxIndex*/, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, std::nullopt, NchwDimensionCount);
        
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[IN_A];
        operatorDesc.OutputTensor = &outputDescs[OUT_Y];
        operatorDesc.OutputScaleTensor = &outputDescs[OUT_Y_SCALE];
        operatorDesc.OutputZeroPointTensor = &outputDescs[OUT_Y_ZERO_POINT];

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(DynamicQuantizeLinear, DmlOperatorDynamicQuantizeLinear);

} // namespace Dml
