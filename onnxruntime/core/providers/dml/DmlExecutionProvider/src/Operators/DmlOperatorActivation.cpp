// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorActivation : public DmlOperator
{
public:
    using Self = DmlOperatorActivation;

    DmlOperatorActivation(
        const MLOperatorKernelCreationContext& kernelCreationContext,
        DML_OPERATOR_TYPE operatorType
        )
    :   DmlOperator(kernelCreationContext)
    {
        // Activation has a single output which is mapped to the first kernel output.  Specifying
        // this manually avoids a problem when activation is used to implement dropout, which may
        // have a 'mask' output which is unused during inference.
        std::vector<std::optional<uint32_t>> kernelOutputIndices = {0};
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, kernelOutputIndices);

        ActivationOperatorDescUnion operatorDesc = {};

        int coerceAxis = TensorAxis::DoNotCoerce;

        switch (operatorType)
        {
        case DML_OPERATOR_ACTIVATION_ELU:
        case DML_OPERATOR_ACTIVATION_CELU:
            operatorDesc.elu.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_SOFTMAX:
        case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
        case DML_OPERATOR_ACTIVATION_HARDMAX:
            {
                const uint32_t onnxDimCount = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0).size());
                coerceAxis = HandleNegativeAxis(kernelCreationContext.GetOptionalAttribute<int>(AttrName::Axis, 1), onnxDimCount);
            }
            break;

        case DML_OPERATOR_ACTIVATION_HARD_SIGMOID:
            operatorDesc.hardSigmoid.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            operatorDesc.hardSigmoid.Beta  = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Beta,  ActivationHelper::GetDefaultBeta(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_LEAKY_RELU:
            operatorDesc.leakyRelu.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_LINEAR:
            operatorDesc.linear.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            operatorDesc.linear.Beta  = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Beta,  ActivationHelper::GetDefaultBeta(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS:
            operatorDesc.parametricSoftplus.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            operatorDesc.parametricSoftplus.Beta  = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Beta,  ActivationHelper::GetDefaultBeta(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_SCALED_ELU:
            operatorDesc.scaledElu.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            operatorDesc.scaledElu.Gamma = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Gamma, 0.0f);
            break;

        case DML_OPERATOR_ACTIVATION_SCALED_TANH:
            operatorDesc.scaledTanh.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            operatorDesc.scaledTanh.Beta  = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Beta,  ActivationHelper::GetDefaultBeta(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_SOFTPLUS:
            operatorDesc.softplus.Steepness = 1.0f;
            break;

        case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU:
            operatorDesc.thresholdedRelu.Alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, ActivationHelper::GetDefaultAlpha(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_SHRINK:
            operatorDesc.shrink.Bias = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Bias, ActivationHelper::GetDefaultBias(operatorType));
            operatorDesc.shrink.Threshold = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Lambda, ActivationHelper::GetDefaultLambda(operatorType));
            break;

        case DML_OPERATOR_ACTIVATION_IDENTITY:
        case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU:
        case DML_OPERATOR_ACTIVATION_RELU:
        case DML_OPERATOR_ACTIVATION_SIGMOID:
        case DML_OPERATOR_ACTIVATION_TANH:
        case DML_OPERATOR_ACTIVATION_SOFTSIGN:
            // No additional parameters to set.
            break;

        default:
            assert(false);
            break;
        }

        if (coerceAxis != TensorAxis::DoNotCoerce)
        {
            m_inputTensorDescs[0] = CreateTensorDescFromInput(kernelCreationContext, 0, coerceAxis);
            m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelCreationContext, 0, coerceAxis);
        }

        gsl::span<const uint32_t> outputSizes = m_outputTensorDescs[0].GetSizes();
        std::vector<DML_TENSOR_DESC> inputDescs;
        std::vector<DML_TENSOR_DESC> outputDescs;

        if (operatorType == DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU)
        {
            // PRelu is unique and accepts its parameters as a second input tensor.

            // The slope tensor is unidirectionally broadcastable. Reshape it based on the desired output sizes.
            m_inputTensorDescs[1] = CreateTensorDescFromInput(kernelCreationContext, 1, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputSizes);

            inputDescs = GetDmlInputDescs();
            outputDescs = GetDmlOutputDescs();
            ML_CHECK_VALID_ARGUMENT(inputDescs.size() == 2);
            ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 1);
            operatorDesc.parameterizedRelu.InputTensor = &inputDescs[0];
            operatorDesc.parameterizedRelu.SlopeTensor = &inputDescs[1];
            operatorDesc.parameterizedRelu.OutputTensor = outputDescs.data();
        }
        else // All other activation descrptions are equivalent to Elu in layout.
        {
            inputDescs = GetDmlInputDescs();
            outputDescs = GetDmlOutputDescs();
            ML_CHECK_VALID_ARGUMENT(inputDescs.size() >= 1);
            ML_CHECK_VALID_ARGUMENT(outputDescs.size() >= 1);
            operatorDesc.elu.InputTensor = inputDescs.data();
            operatorDesc.elu.OutputTensor = outputDescs.data();
        }

        DML_OPERATOR_DESC opDesc = { operatorType, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

// A specific type of operation for registration.
template <DML_OPERATOR_TYPE OperatorType>
class DmlOperatorActivationTemplate : public DmlOperatorActivation
{
public:
    DmlOperatorActivationTemplate(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperatorActivation(kernelCreationContext, OperatorType)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Sigmoid,             DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SIGMOID>);
DML_OP_DEFINE_CREATION_FUNCTION(HardSigmoid,         DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_HARD_SIGMOID>);
DML_OP_DEFINE_CREATION_FUNCTION(Tanh,                DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_TANH>);
DML_OP_DEFINE_CREATION_FUNCTION(ScaledTanh,          DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SCALED_TANH>);
DML_OP_DEFINE_CREATION_FUNCTION(Relu,                DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Celu,                DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_CELU>);
DML_OP_DEFINE_CREATION_FUNCTION(LeakyRelu,           DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_LEAKY_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(PRelu,               DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(ThresholdedRelu,     DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Elu,                 DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_ELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Selu,                DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SCALED_ELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Softsign,            DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SOFTSIGN>);
DML_OP_DEFINE_CREATION_FUNCTION(Softplus,            DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SOFTPLUS>);
DML_OP_DEFINE_CREATION_FUNCTION(ParametricSoftplus,  DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS>);
DML_OP_DEFINE_CREATION_FUNCTION(Dropout,             DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_IDENTITY>);
DML_OP_DEFINE_CREATION_FUNCTION(Softmax,             DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SOFTMAX>);
DML_OP_DEFINE_CREATION_FUNCTION(LogSoftmax,          DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_LOG_SOFTMAX>);
DML_OP_DEFINE_CREATION_FUNCTION(Hardmax,             DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_HARDMAX>);
DML_OP_DEFINE_CREATION_FUNCTION(Shrink,              DmlOperatorActivationTemplate<DML_OPERATOR_ACTIVATION_SHRINK>);

} // namespace Dml
