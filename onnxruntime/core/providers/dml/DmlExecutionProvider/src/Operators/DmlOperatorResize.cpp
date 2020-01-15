// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorResize : public DmlOperator, public ResizeHelper
{
public:
    // Resample a multidimensional image to a new size.
    DmlOperatorResize(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        ResizeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(!m_scales.empty(), "Resize/Upsample expect scales, either a 2nd input tensors or 'scales' attribute.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Resize/Upsample expect 1 output tensor.");

        // Use only the first input tensor. In the case of Resize or the later Upsample-v9,
        // the second tensor is CPU based and should not be passed to Resize.
        std::vector<std::optional<uint32_t>> inputIndices = { 0 };
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        // Because DirectML supports a limited number of dimensions, try to squeeze the dimension count
        // to only those which actually matter. Models sometimes use a greater number of dimensions,
        // even though those dimensions have no significance and can be elided (nop 1's), coercing the
        // total dimension count back down to a supported value.

        std::vector<uint32_t> squeezedInputShape = m_inputDimensions;
        std::vector<uint32_t> squeezedOutputShape = m_outputDimensions;
        std::vector<uint32_t> squeezableDimensionIndices;
        std::vector<float> paddedScales = m_scales;
        FindValueIndices<uint32_t>(gsl::make_span(m_outputDimensions), 1u, /*out*/ squeezableDimensionIndices);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ squeezedInputShape);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ paddedScales);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ squeezedOutputShape);

        // Update the tensor descriptions.
        MLOperatorTensorDataType inputTensorDataType = kernelCreationContext.GetInputEdgeDescription(0).tensorDataType;
        auto inputTensorDesc = TensorDesc(inputTensorDataType, squeezedInputShape, squeezedInputShape, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, NchwDimensionCount, 0);
        auto outputTensorDesc = TensorDesc(inputTensorDataType, squeezedOutputShape, squeezedOutputShape, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, NchwDimensionCount, 0);
        m_inputTensorDescs[0] = inputTensorDesc;
        m_outputTensorDescs[0] = outputTensorDesc;

        // If the output tensor dimension count was right-aligned to a larger size,
        // then ensure that scales has the same count as the tensor rank by inserting
        // leading ones, since DirectML requires the scales to have the same count.
        const uint32_t squeezedDimCount = gsl::narrow_cast<uint32_t>(squeezedOutputShape.size());
        const uint32_t dmlCompatibleDimCount = outputTensorDesc.GetDimensionCount();
        if (dmlCompatibleDimCount > squeezedDimCount)
        {
            paddedScales.insert(paddedScales.begin(), dmlCompatibleDimCount - squeezedDimCount, 1.0f);
        }

        std::string mode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::Mode, "NEAREST");
        DML_INTERPOLATION_MODE interpolationMode = Dml::MapStringToInteropolationMode(mode);

        // Create the operator description.
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_RESAMPLE_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.InterpolationMode = interpolationMode;
        operatorDesc.Scales = paddedScales.data();
        operatorDesc.ScaleCount = gsl::narrow_cast<uint32_t>(paddedScales.size());

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_RESAMPLE, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Resize, DmlOperatorResize);
DML_OP_DEFINE_CREATION_FUNCTION(Upsample, DmlOperatorResize);

} // namespace Dml
