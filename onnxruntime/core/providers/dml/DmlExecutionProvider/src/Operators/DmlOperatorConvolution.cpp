// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorConvolution : public DmlOperator, public ConvolutionHelperBase
{
public:
    using Self = DmlOperatorConvolution;

    DmlOperatorConvolution(
        const MLOperatorKernelCreationContext& kernelInfo,
        DML_CONVOLUTION_MODE mode,
        DML_CONVOLUTION_DIRECTION direction,
        bool hasDynamicPads,
        bool isNhwc
        )
    :   DmlOperator(kernelInfo),
        ConvolutionHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), direction == DML_CONVOLUTION_DIRECTION_BACKWARD, hasDynamicPads, isNhwc, 0, 1)
    {
        uint32_t biasIndex = hasDynamicPads ? 3 : 2;
        bool hasBiasInput = kernelInfo.GetInputCount() > biasIndex;

        std::vector<std::optional<uint32_t>> kernelInputIndices = { 0, 1, biasIndex };

        DmlOperator::Initialize(kernelInfo, kernelInputIndices, std::nullopt, std::nullopt, std::nullopt, NchwDimensionCount);

        // Vibranium DirectML is limited to handle only 2D and 3D convolution (4D and 5D tensors). So for 1D tensors,
        // massage the tensor descriptions. By default, the TensorDesc simply right aligns all the values up to 4D
        // (padding the leading dimensions with 1's), but 1D tensors actually need to insert the 1 between C and W.
        // e.g. [2,3,4] becomes [2,3,1,4]
        m_inputTensorDescs[0] = CreateTensorDescFromInput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt, NchwDimensionCount);
        m_inputTensorDescs[1] = CreateTensorDescFromInput(kernelInfo, 1, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt, NchwDimensionCount);
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::NoPlacementAdjustment, NonspatialDimensionCount, std::nullopt, NchwDimensionCount);

        if (isNhwc)
        {
            // Restrict to 4D like other implementations
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[0].GetDimensionCount() == 4);
            const auto inputSizes = m_inputTensorDescs[0].GetSizes();
            const uint32_t inputBatch = inputSizes[0];
            const uint32_t inputHeight = inputSizes[1];
            const uint32_t inputWidth = inputSizes[2];
            const uint32_t inputChannels = inputSizes[3];
            const std::array<uint32_t, 4> nchwInputSizes = {inputBatch, inputChannels, inputHeight, inputWidth};
            const std::array<uint32_t, 4> nchwInputStrides = {inputHeight * inputWidth * inputChannels, 1, inputWidth * inputChannels, inputChannels};
            m_inputTensorDescs[0] = TensorDesc(m_inputTensorDescs[0].GetDmlDataType(), nchwInputSizes, nchwInputStrides);

            // Restrict to 4D like other implementations
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[1].GetDimensionCount() == 4);
            const auto weightSizes = m_inputTensorDescs[1].GetSizes();
            const uint32_t featureMaps = weightSizes[0];
            const uint32_t kernelHeight = weightSizes[1];
            const uint32_t kernelWidth = weightSizes[2];
            const uint32_t channelsPerGroup = weightSizes[3];
            const std::array<uint32_t, 4> nchwKernelSizes = {featureMaps, channelsPerGroup, kernelHeight, kernelWidth};
            const std::array<uint32_t, 4> nchwKernelStrides = {kernelHeight * kernelWidth * channelsPerGroup, 1, kernelWidth * channelsPerGroup, channelsPerGroup};
            m_inputTensorDescs[1] = TensorDesc(m_inputTensorDescs[1].GetDmlDataType(), nchwKernelSizes, nchwKernelStrides);

            // Restrict to 4D like other implementations
            ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs[0].GetDimensionCount() == 4);
            const auto outputSizes = m_outputTensorDescs[0].GetSizes();
            const uint32_t outputBatch = outputSizes[0];
            const uint32_t outputHeight = outputSizes[1];
            const uint32_t outputWidth = outputSizes[2];
            const uint32_t outputChannels = outputSizes[3];
            const std::array<uint32_t, 4> nchwOutputSizes = {outputBatch, outputChannels, outputHeight, outputWidth};
            const std::array<uint32_t, 4> nchwOutputStrides = {outputHeight * outputWidth * outputChannels, 1, outputWidth * outputChannels, outputChannels};
            m_outputTensorDescs[0] = TensorDesc(m_outputTensorDescs[0].GetDmlDataType(), nchwOutputSizes, nchwOutputStrides);
        }

        // Bias is optional so only adjust it if it exists.
        if (hasBiasInput)
        {
            uint32_t inputDimSize = kernelInfo.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
            ML_CHECK_VALID_ARGUMENT(
                inputDimSize >= 3 && inputDimSize <= 5,
                "Bias can only be used with 3D/4D/5D tensors."
                );
            uint32_t dmlDimSize = m_inputTensorDescs[0].GetDimensionCount();

            // Resize the bias to be the same dimension as the input tensor.
            // The 1D tensor needs to be moved to the C channel.
            m_inputTensorDescs[biasIndex] = CreateTensorDescFromInput(
                kernelInfo,
                biasIndex,
                TensorAxis::DoNotCoerce,
                TensorAxis::C,
                TensorAxis::LeftAligned,
                std::nullopt,
                dmlDimSize
                );
        }

        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // Form transient kernel arguments with spatial dimensions padded up to at least 2,
        // since the DirectML API rejects 1D convolution. Leave the base m_kernel alone
        // so that all output tensor size computations are correct.
        KernelArgs kernelArgs(m_kernel, NchwSpatialDimensionCount);

        // Zero the output padding before sending to DirectML. Although it was needed to compute
        // the output size, we don't want DML to see the values, which should just be ignored.
        memset(kernelArgs.outputPadding, 0, sizeof(kernelArgs.outputPadding));

        DML_CONVOLUTION_OPERATOR_DESC convDesc = {};
        convDesc.InputTensor = &inputDescs[0];
        convDesc.FilterTensor = &inputDescs[1];
        convDesc.BiasTensor = hasBiasInput ? &inputDescs[biasIndex] : nullptr;
        convDesc.OutputTensor = &outputDescs[0];
        convDesc.Mode = mode;
        convDesc.Direction = direction;
        convDesc.DimensionCount = kernelArgs.spatialDimensionCount;
        convDesc.Strides = kernelArgs.strides;
        convDesc.Dilations = kernelArgs.dilations;
        convDesc.StartPadding = kernelArgs.startPadding;
        convDesc.EndPadding = kernelArgs.endPadding;
        convDesc.OutputPadding = kernelArgs.outputPadding;
        convDesc.GroupCount = m_groupCount;
        convDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CONVOLUTION, &convDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

// A specific type of operation for registration.
template <DML_CONVOLUTION_MODE Mode, DML_CONVOLUTION_DIRECTION Direction, bool hasDynamicPads = false, bool isNhwc = false>
class DmlOperatorConvolutionTemplate : public DmlOperatorConvolution
{
public:
    DmlOperatorConvolutionTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorConvolution(kernelInfo, Mode, Direction, hasDynamicPads, isNhwc)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Conv,                           DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(NhwcConv,                       DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD, false, true>);
DML_OP_DEFINE_CREATION_FUNCTION(ConvTranspose,                  DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedConv,                   DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedConvTranspose,          DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(ConvTransposeWithDynamicPads,   DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD, true>);

} // namespace Dml
