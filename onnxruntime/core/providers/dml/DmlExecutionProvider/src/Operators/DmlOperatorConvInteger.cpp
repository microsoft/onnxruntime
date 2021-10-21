// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorConvInteger : public DmlOperator, public ConvolutionHelperBase
{
private:
    enum InputTensors 
    { 
        IN_X, 
        IN_X_ZERO_POINT, 
        IN_F, 
        IN_F_ZERO_POINT, 
    };
    
public:
    using Self = DmlOperatorConvInteger;

    DmlOperatorConvInteger(
        const MLOperatorKernelCreationContext& kernelInfo
        )
    :   DmlOperator(kernelInfo),
        ConvolutionHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), false, false, 0, 1)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 2, 1, 3};
        std::vector<std::optional<uint32_t>> kernelOutputIndices = {0};

        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

        // DirectML is limited to handle only 2D. So for 1D tensors, massage the tensor descriptions. By default, the 
        // TensorDesc simply right aligns all the values up to 4D (padding the leading dimensions with 1's), 
        // but 1D tensors actually need to insert the 1 between C and W. e.g. [2,3,4] becomes [2,3,1,4]
        m_inputTensorDescs[IN_X] = CreateTensorDescFromInput(kernelInfo, 0/*Onnx Index*/, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt);
        m_inputTensorDescs[IN_F] = CreateTensorDescFromInput(kernelInfo, 1/*Onnx Index*/, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt);

        uint32_t dmlDimSize = m_inputTensorDescs[0].GetDimensionCount();

        // Resize the Filter ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the C channel.
        m_inputTensorDescs[IN_F_ZERO_POINT] = CreateTensorDescFromInput(
            kernelInfo, 
            3/*Onnx Index*/, 
            TensorAxis::DoNotCoerce, 
            TensorAxis::C,
            TensorAxis::LeftAligned,
            std::nullopt,
            dmlDimSize
            );

        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // Form transient kernel arguments with spatial dimensions padded up to at least 2,
        // since the DirectML API rejects 1D convolution. Leave the base m_kernel alone
        // so that all output tensor size computations are correct.
        KernelArgs kernelArgs(m_kernel, NchwSpatialDimensionCount);

        DML_CONVOLUTION_INTEGER_OPERATOR_DESC convDesc = {};
        convDesc.InputTensor = &inputDescs[IN_X];
        convDesc.InputZeroPointTensor = inputDescs[IN_X_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_X_ZERO_POINT] : nullptr;
        convDesc.FilterTensor = &inputDescs[IN_F];
        convDesc.FilterZeroPointTensor = inputDescs[IN_F_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_F_ZERO_POINT] : nullptr;
        convDesc.OutputTensor = &outputDescs[0];
        convDesc.DimensionCount = kernelArgs.spatialDimensionCount;
        convDesc.Strides = kernelArgs.strides;
        convDesc.Dilations = kernelArgs.dilations;
        convDesc.StartPadding = kernelArgs.startPadding;
        convDesc.EndPadding = kernelArgs.endPadding;
        convDesc.GroupCount = m_groupCount;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CONVOLUTION_INTEGER, &convDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ConvInteger, DmlOperatorConvInteger);

} // namespace Dml
