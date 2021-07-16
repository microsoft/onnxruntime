// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorRegionOfInterestAlign : public DmlOperator, public RoiAlignHelper
{
public:
    using Self = DmlOperatorRegionOfInterestAlign;

    DmlOperatorRegionOfInterestAlign(const MLOperatorKernelCreationContext& kernelCreationContext, uint32_t opsetVersion)
    :   DmlOperator(kernelCreationContext),
        RoiAlignHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription(), opsetVersion)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3, "RoiAlign expects 3 input tensors.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "RoiAlign expects 1 output tensor.");

        DmlOperator::Initialize(kernelCreationContext);
        DmlOperator::Remap64bitDmlDataTypesTo32bit();
        m_inputTensorDescs[2].ForceUnsignedDataType();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        constexpr NameAndIndex mapping[] =
        {
            {"max", DML_REDUCE_FUNCTION_MAX},
            {"avg", DML_REDUCE_FUNCTION_AVERAGE},
        };
        const std::string mode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::Mode, "avg");
        const auto optionalReductionFunction = TryMapStringToIndex<DML_REDUCE_FUNCTION>(mode, mapping);
        const float spatialScale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::SpatialScale, 1.0f);
        const int32_t samplesPerOutput = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::SamplingRatio, 0u);
        ML_CHECK_VALID_ARGUMENT(samplesPerOutput >= 0, "sampling_ratio must be 0 or positive.");
        ML_CHECK_VALID_ARGUMENT(!!optionalReductionFunction, "Unsupported RoiAlign mode.");

        DML_ROI_ALIGN_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ROITensor = &inputDescs[1];
        operatorDesc.BatchIndicesTensor = &inputDescs[2];
        operatorDesc.OutputTensor = &outputDescs[0];
        operatorDesc.SpatialScaleX = spatialScale; // ONNX uses the same scale for X and Y.
        operatorDesc.SpatialScaleY = spatialScale;
        operatorDesc.OutOfBoundsInputValue = 0.0f; // ONNX does not specify a value for input elements outside bounds.
        operatorDesc.MinimumSamplesPerOutput = (samplesPerOutput == 0) ? 1          : samplesPerOutput;
        operatorDesc.MaximumSamplesPerOutput = (samplesPerOutput == 0) ? UINT32_MAX : samplesPerOutput;
        operatorDesc.ReductionFunction = *optionalReductionFunction;
        operatorDesc.InterpolationMode = DML_INTERPOLATION_MODE_LINEAR;
        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ROI_ALIGN, &operatorDesc };

        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(RoiAlign10, VersionedKernel<DmlOperatorRegionOfInterestAlign, 10>);

} // namespace Dml
