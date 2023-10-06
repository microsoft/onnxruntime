// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorQLinearConv : public DmlOperator, public ConvolutionHelperBase
{
private:
    enum InputTensors
    {
        IN_X,
        IN_X_SCALE,
        IN_X_ZERO_POINT,
        IN_F,
        IN_F_SCALE,
        IN_F_ZERO_POINT,
        IN_BIAS,
        IN_Y_SCALE,
        IN_Y_ZERO_POINT
    };

public:
    using Self = DmlOperatorQLinearConv;

    DmlOperatorQLinearConv(
        const MLOperatorKernelCreationContext& kernelInfo
        )
    :   DmlOperator(kernelInfo),
        ConvolutionHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), false, false, false, 0, 3)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2, 3, 4, 5, 8, 6, 7};
        std::vector<std::optional<uint32_t>> kernelOutputIndices = {0};

        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

        // DirectML is limited to handle only 2D. So for 1D tensors, massage the tensor descriptions. By default, the
        // TensorDesc simply right aligns all the values up to 4D (padding the leading dimensions with 1's),
        // but 1D tensors actually need to insert the 1 between C and W. e.g. [2,3,4] becomes [2,3,1,4]
        m_inputTensorDescs[IN_X] = CreateTensorDescFromInput(kernelInfo, 0/*Onnx Index*/, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt);
        m_inputTensorDescs[IN_F] = CreateTensorDescFromInput(kernelInfo, 3/*Onnx Index*/, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt);

        uint32_t inputDimSize = kernelInfo.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
        ML_CHECK_VALID_ARGUMENT(
            inputDimSize >= 3 && inputDimSize <= 4,
            "Input can only be used with 3D/4D tensors."
            );

        uint32_t dmlDimSize = m_inputTensorDescs[0].GetDimensionCount();

        // Bias is optional so only adjust it if it exists.
        if (m_inputTensorDescs[IN_BIAS].GetDmlDesc().Desc != nullptr)
        {
            // Resize the bias to be the same dimension as the input tensor.
            // The 1D tensor needs to be moved to the C channel.
            m_inputTensorDescs[IN_BIAS] = CreateTensorDescFromInput(
                kernelInfo,
                8/*Onnx Index*/,
                TensorAxis::DoNotCoerce,
                TensorAxis::C,
                TensorAxis::LeftAligned,
                std::nullopt,
                dmlDimSize
                );
        }

        // Resize the Filter ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the C channel.
        m_inputTensorDescs[IN_F_ZERO_POINT] = CreateTensorDescFromInput(
            kernelInfo,
            5/*Onnx Index*/,
            TensorAxis::DoNotCoerce,
            TensorAxis::C,
            TensorAxis::LeftAligned,
            std::nullopt,
            dmlDimSize
            );
        // Resize the Filter Scale to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the C channel.
        m_inputTensorDescs[IN_F_SCALE] = CreateTensorDescFromInput(
            kernelInfo,
            4/*Onnx Index*/,
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

        DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC convDesc = {};
        convDesc.InputTensor = &inputDescs[IN_X];
        convDesc.InputScaleTensor = &inputDescs[IN_X_SCALE];
        convDesc.InputZeroPointTensor = inputDescs[IN_X_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_X_ZERO_POINT] : nullptr;
        convDesc.FilterTensor = &inputDescs[IN_F];
        convDesc.FilterScaleTensor = &inputDescs[IN_F_SCALE];
        convDesc.FilterZeroPointTensor = inputDescs[IN_F_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_F_ZERO_POINT] : nullptr;
        convDesc.BiasTensor = inputDescs[IN_BIAS].Desc != nullptr ? &inputDescs[IN_BIAS] : nullptr;
        convDesc.OutputScaleTensor = &inputDescs[IN_Y_SCALE];
        convDesc.OutputZeroPointTensor = inputDescs[IN_Y_ZERO_POINT].Desc != nullptr ? &inputDescs[IN_Y_ZERO_POINT] : nullptr;
        convDesc.OutputTensor = &outputDescs[0];
        convDesc.DimensionCount = kernelArgs.spatialDimensionCount;
        convDesc.Strides = kernelArgs.strides;
        convDesc.Dilations = kernelArgs.dilations;
        convDesc.StartPadding = kernelArgs.startPadding;
        convDesc.EndPadding = kernelArgs.endPadding;
        convDesc.GroupCount = m_groupCount;

        TryConvertTensorToBroadcastScalar(kernelInfo, convDesc.InputScaleTensor,      IN_X_SCALE);
        TryConvertTensorToBroadcastScalar(kernelInfo, convDesc.InputZeroPointTensor,  IN_X_ZERO_POINT);

        TryConvertTensorToBroadcastScalar(kernelInfo, convDesc.FilterScaleTensor,     IN_F_SCALE);
        TryConvertTensorToBroadcastScalar(kernelInfo, convDesc.FilterZeroPointTensor, IN_F_ZERO_POINT);

        TryConvertTensorToBroadcastScalar(kernelInfo, convDesc.OutputScaleTensor,     IN_Y_SCALE);
        TryConvertTensorToBroadcastScalar(kernelInfo, convDesc.OutputZeroPointTensor, IN_Y_ZERO_POINT);

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION, &convDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(QLinearConv,                    DmlOperatorQLinearConv);

} // namespace Dml
