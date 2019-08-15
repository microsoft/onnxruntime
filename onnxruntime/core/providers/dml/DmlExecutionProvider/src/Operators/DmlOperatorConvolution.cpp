//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
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
        DML_CONVOLUTION_DIRECTION direction
        )
    :   DmlOperator(kernelInfo),
        ConvolutionHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), direction == DML_CONVOLUTION_DIRECTION_BACKWARD)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 2);

        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

        // Bias is optional so only adjust it if it exists.
        if (kernelInfo.GetInputCount() > 2)
        {
            uint32_t inputDimSize = kernelInfo.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
            ML_CHECK_VALID_ARGUMENT(
                inputDimSize == NcdhwDimensionCount || inputDimSize == NchwDimensionCount,
                "Bias can only be used with 4D or 5D tensors."
                );

            // Resize the bias to be the same dimension as the input tensor
            m_inputTensorDescs[2] = CreateTensorDescFromInput(
                kernelInfo, 
                2, 
                TensorAxis::DoNotCoerce, 
                TensorAxis::C, 
                std::nullopt,
                inputDimSize
                );
        }

        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_CONVOLUTION_OPERATOR_DESC convDesc = {};
        convDesc.InputTensor = &inputDescs[0];
        convDesc.FilterTensor = &inputDescs[1];
        convDesc.BiasTensor = kernelInfo.GetInputCount() > 2 ? &inputDescs[2] : nullptr;
        convDesc.OutputTensor = &outputDescs[0];
        convDesc.Mode = mode;
        convDesc.Direction = direction;
        convDesc.DimensionCount = m_kernel.spatialDimensionCount;
        convDesc.Strides = m_kernel.strides;
        convDesc.Dilations = m_kernel.dilations;
        convDesc.StartPadding = m_kernel.startPadding;
        convDesc.EndPadding = m_kernel.endPadding;
        convDesc.OutputPadding = m_kernel.outputPadding;
        convDesc.GroupCount = m_groupCount;
        convDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CONVOLUTION, &convDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

// A specific type of operation for registration.
template <DML_CONVOLUTION_MODE Mode, DML_CONVOLUTION_DIRECTION Direction>
class DmlOperatorConvolutionTemplate : public DmlOperatorConvolution
{
public:
    DmlOperatorConvolutionTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorConvolution(kernelInfo, Mode, Direction)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Conv,               DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(ConvTranspose,      DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(FusedConv,          DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD>);
DML_OP_DEFINE_CREATION_FUNCTION(FusedConvTranspose, DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD>);

} // namespace Dml
