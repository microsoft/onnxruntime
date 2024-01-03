// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorMatMulIntegerToFloat : public DmlOperator
{
    enum OrtInputTensors : uint32_t
    {
        ortA,
        ortB,
        ortAScale,
        ortBScale,
        ortAZeroPoint,
        ortBZeroPoint,
        ortBias,
        ortInputCount
    };
    
    enum DmlInputIndex : uint32_t
    {
        dmlA,
        dmlAScale,
        dmlAZeroPoint,
        dmlB,
        dmlBScale,
        dmlBZeroPoint,
        dmlBias,
        dmlInputCount,
    };

public:
    DmlOperatorMatMulIntegerToFloat(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo)
    {
        std::vector<std::optional<uint32_t>> inputIndices = { OrtInputTensors::ortA, OrtInputTensors::ortAScale, OrtInputTensors::ortAZeroPoint, OrtInputTensors::ortB, OrtInputTensors::ortBScale, OrtInputTensors::ortBZeroPoint, OrtInputTensors::ortBias };
        DmlOperator::Initialize(kernelInfo, inputIndices);

        std::vector<DimensionType> inputShape0 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(OrtInputTensors::ortA);
        std::vector<DimensionType> inputShape1 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(OrtInputTensors::ortB);
        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        OperatorHelper::MatMulShapeMapping(inputShape0, inputShape1, outputShape);

        // Initialize the input descriptions with broadcasting
        m_inputTensorDescs[DmlInputIndex::dmlA] = CreateTensorDescFromInput(kernelInfo, OrtInputTensors::ortA, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape0);
        m_inputTensorDescs[DmlInputIndex::dmlB] = CreateTensorDescFromInput(kernelInfo, OrtInputTensors::ortB, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, inputShape1);

        // Broadcast Bias tensor to the shape of the output tensor.
        if(kernelInfo.IsInputValid(OrtInputTensors::ortBias)) {
            
            m_inputTensorDescs[DmlInputIndex::dmlBias] = CreateTensorDescFromInput(kernelInfo, OrtInputTensors::ortBias, TensorAxis::DoNotCoerce,
                TensorAxis::W, TensorAxis::RightAligned, outputShape);
        }

        uint32_t dmlDimSize = m_inputTensorDescs[DmlInputIndex::dmlA].GetDimensionCount();
        // Resize the A Scale to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        m_inputTensorDescs[DmlInputIndex::dmlAScale] = CreateTensorDescFromInput(
            kernelInfo, 
            OrtInputTensors::ortAScale,
            TensorAxis::DoNotCoerce, 
            TensorAxis::H,
            TensorAxis::LeftAligned,
            std::nullopt,
            dmlDimSize
            );

        // Resize the A ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        if (kernelInfo.IsInputValid(OrtInputTensors::ortAZeroPoint))
        {

            m_inputTensorDescs[DmlInputIndex::dmlAZeroPoint] = CreateTensorDescFromInput(
                kernelInfo, 
                OrtInputTensors::ortAZeroPoint,
                TensorAxis::DoNotCoerce, 
                TensorAxis::H,
                TensorAxis::LeftAligned,
                std::nullopt,
                dmlDimSize
                );
        }

        // B Zeropoint and BScale are already aligned in the W dimension so no need to align them

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC matMulDesc = {};
        matMulDesc.ATensor = &inputDescs[DmlInputIndex::dmlA];
        matMulDesc.AScaleTensor = &inputDescs[DmlInputIndex::dmlAScale];
        matMulDesc.AZeroPointTensor = inputDescs[DmlInputIndex::dmlAZeroPoint].Desc != nullptr ? &inputDescs[DmlInputIndex::dmlAZeroPoint] : nullptr;
        matMulDesc.BTensor = &inputDescs[DmlInputIndex::dmlB];
        matMulDesc.BScaleTensor = &inputDescs[DmlInputIndex::dmlBScale];
        matMulDesc.BZeroPointTensor = inputDescs[DmlInputIndex::dmlBZeroPoint].Desc != nullptr ? &inputDescs[DmlInputIndex::dmlBZeroPoint] : nullptr;
        matMulDesc.BiasTensor = inputDescs[DmlInputIndex::dmlBias].Desc != nullptr ? &inputDescs[DmlInputIndex::dmlBias] : nullptr;
        matMulDesc.OutputTensor = &outputDescs[0];

        DML_OPERATOR_DESC opDesc = { (DML_OPERATOR_TYPE) DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT, &matMulDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MatMulIntegerToFloat, DmlOperatorMatMulIntegerToFloat);

} // namespace Dml