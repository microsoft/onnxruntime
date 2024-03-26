// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

constexpr NameAndIndex coordinateTransformationModes[] =
{
    {"half_pixel", 0},
    {"half_pixel_symmetric", 1},
    {"pytorch_half_pixel", 2},
    {"align_corners", 3},
    {"asymmetric", 4},
    {"tf_half_pixel_for_nn", 5},
    {"tf_crop_and_resize", 6},
};

constexpr NameAndIndex nearestNeighborRoundingModes[] =
{
    {"", 0},
    {"round_prefer_floor", 0},  // round halves down
    {"round_prefer_ceil", 1},   // round halves up
    {"floor", 2},               // round always down
    {"ceil", 3},                // round always up
};

void ComputePixelOffsetsAndScales(
    const MLOperatorKernelCreationContext& kernelCreationContext,
    gsl::span<const float> regionOfInterest, // May be empty depending on mode.
    gsl::span<const uint32_t> inputDimensions,
    gsl::span<const uint32_t> outputDimensions,
    /*inout*/ gsl::span<float> scales,
    /*out*/   gsl::span<float> inputPixelOffsets,
    /*out*/   gsl::span<float> outputPixelOffsets
    )
{
    assert(inputDimensions.size() == outputDimensions.size());
    assert(inputPixelOffsets.size() == outputPixelOffsets.size());
    assert(inputPixelOffsets.size() == scales.size());
    assert(inputPixelOffsets.size() == inputDimensions.size());
    assert(regionOfInterest.empty() || regionOfInterest.size() == inputDimensions.size() * 2);

    std::string coordinateTransformationMode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::CoordinateTransformationMode, "half_pixel");
    auto optionalCoordinateTransformationModeValue = TryMapStringToIndex(coordinateTransformationMode, coordinateTransformationModes);
    if (!optionalCoordinateTransformationModeValue)
    {
        ML_INVALID_ARGUMENT("Unsupported 'coordinate_transformation_mode'");
    }
    uint32_t coordinateTransformationModeValue = *optionalCoordinateTransformationModeValue;

    ML_CHECK_VALID_ARGUMENT(
        !regionOfInterest.empty() || coordinateTransformationModeValue != 6 /*tf_crop_and_resize*/,
        "Resize expects 'roi' tensor for 'tf_crop_and_resize' mode."
    );

    const uint32_t rank = gsl::narrow_cast<uint32_t>(inputDimensions.size());

    // Fill in all the input/output pixel offset for each axis,
    // and recompute the scale for certain modes.

    for (uint64_t i = 0; i < rank; ++i)
    {
        float inputPixelOffset = 0;
        float outputPixelOffset = 0;

        // All these mapping modes can be generalized to the equations:
        //
        // output_coordinate = (input_coordinate  + input_offset ) * scale + output_offset
        // input_coordinate  = (output_coordinate - output_offset) / scale - input_offset
        //
        // With DML, a scale > 1 maps input to an upsampled output, and a positive pixel
        // offset shifts the input contents to the right/down in the output.
        //
        // Since the equations from ONNX are in terms of mapping the output coordinate back
        // to the input coordinate, any offsets need their signs flipped. e.g. For "half_pixel",
        // the "x_resized" is the output coordinate, and the "+ 0.5" is the output coordinate
        // adjustment which needs to be -0.5 when passed to DML.

        switch (coordinateTransformationModeValue)
        {
        case 0:
            // coordinate_transformation_mode is "half_pixel",
            // x_original = (x_resized + 0.5) / scale - 0.5
            inputPixelOffset = 0.5;
            outputPixelOffset = -0.5;
            // Keep existing scales.
            break;

        case 1:
            // coordinate_transformation_mode is "half_pixel_symmetric",
            // adjustment = output_width_int / output_width
            // center = input_width / 2
            // offset = center * (1 - adjustment)
            // x_original = (x + 0.5) / scale - (0.5 - offset)
            // x_original = (x + 0.5) / scale - (0.5 - [(input_width / 2) * (1 - (output_width_int / output_width))])
            // output_width can be fractional when calculated with scale factor
            inputPixelOffset = 0.5f - float((inputDimensions[i] / 2.0f) * (1.0f - outputDimensions[i] / (scales[i] * inputDimensions[i])));
            outputPixelOffset = -0.5;
            break;

        case 2:
            // if coordinate_transformation_mode is "pytorch_half_pixel",
            // x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0
            if (inputDimensions[i] <= 1)
            {
                inputPixelOffset = 0.0;
                outputPixelOffset = 0.0;
                scales[i] = FLT_MAX; // Set large scale so all output pixels map to 0th input pixel.
            }
            else
            {
                inputPixelOffset = 0.5;
                outputPixelOffset = -0.5;
                // Keep existing scales.
            }
            break;

        case 3:
            // if coordinate_transformation_mode is "align_corners",
            // x_original = x_resized * (length_original - 1) / (length_resized - 1)
            inputPixelOffset = 0.0;
            outputPixelOffset = 0.0;
            if (outputDimensions[i] <= 1 || inputDimensions[i] <= 1)
            {
                // Protect against division by zero when either input/output is a single pixel.
                scales[i] = FLT_MAX;
            }
            else
            {
                // Recalcalculate scale, ignoring existing one (only used to determine output size).
                scales[i] = float(outputDimensions[i] - 1) / (inputDimensions[i] - 1);
            }
            break;

        case 4:
            // if coordinate_transformation_mode is "asymmetric",
            // x_original = x_resized / scale
            inputPixelOffset = 0.0;
            outputPixelOffset = 0.0;
            // Keep existing scales.
            break;

        case 5:
            // if coordinate_transformation_mode is "tf_half_pixel_for_nn",
            // x_original = (x_resized + 0.5) / scale
            inputPixelOffset = 0.0;
            outputPixelOffset = -0.5;
            // Keep existing scales.
            break;

        case 6:
            // if coordinate_transformation_mode is "tf_crop_and_resize",
            // x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1)
            //                                 : 0.5 * (start_x + end_x) * (length_original - 1)
            if (inputDimensions[i] > 1)
            {
                assert(regionOfInterest.size() == rank * 2);

                // Fold this part of the equation into the input offset: start_x * (length_original - 1)
                inputPixelOffset = -(regionOfInterest[i] * (inputDimensions[i] - 1));
                outputPixelOffset = 0.0;

                // Fold this part to scale: (end_x - start_x) * (length_original - 1) / (length_resized - 1)
                float computedScale = float(outputDimensions[i] - 1)
                                    / std::max((regionOfInterest[i + rank] - regionOfInterest[i]) * (inputDimensions[i] - 1), 1.0f);
                scales[i] = computedScale;
            }
            else // inputDimensions[i] <= 1
            {
                // 0.5 * (start_x + end_x) * (length_original - 1)
                inputPixelOffset = -0.5f * (regionOfInterest[i] + regionOfInterest[i + rank]) * (inputDimensions[i] - 1);
                outputPixelOffset = 0.0;
                scales[i] = 1;
            }
            break;

        default:
            assert(false); // TryMapStringToIndex would have already bailed above.
        }

        inputPixelOffsets[i] = inputPixelOffset;
        outputPixelOffsets[i] = outputPixelOffset;
    }
}

class DmlOperatorResize : public DmlOperator, public ResizeHelper
{
public:
    // Resample a multidimensional image to a new size.
    DmlOperatorResize(const MLOperatorKernelCreationContext& kernelCreationContext, uint32_t opsetVersion)
    :   DmlOperator(kernelCreationContext),
        ResizeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription(), opsetVersion)
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
        std::vector<float> inputPixelOffsets(paddedScales.size());
        std::vector<float> outputPixelOffsets(paddedScales.size());

        ComputePixelOffsetsAndScales(
            kernelCreationContext,
            m_regionOfInterest, // May be empty depending on mode.
            m_inputDimensions,
            m_outputDimensions,
            /*inout*/ paddedScales,
            /*out*/ inputPixelOffsets,
            /*out*/ outputPixelOffsets
        );

        // Find any useless dimensions of size 1 that occur in both input and output.
        // This enables higher dimension cases (where models prepend unnecessary
        // dimensions) beyond DML's supported dimension count of 4.
        for (size_t i = 0, rank = m_outputDimensions.size(); i < rank; ++i)
        {
            if (m_inputDimensions[i] == 1 && m_outputDimensions[i] == 1)
            {
                squeezableDimensionIndices.push_back(gsl::narrow_cast<uint32_t>(i));
            }
        }

        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ squeezedInputShape);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ paddedScales);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ inputPixelOffsets);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ outputPixelOffsets);
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
            inputPixelOffsets.insert(inputPixelOffsets.begin(), dmlCompatibleDimCount - squeezedDimCount, 0.5f);
            outputPixelOffsets.insert(outputPixelOffsets.begin(), dmlCompatibleDimCount - squeezedDimCount, -0.5f);
        }

        std::string mode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::Mode, "NEAREST");
        DML_INTERPOLATION_MODE interpolationMode = Dml::MapStringToInteropolationMode(mode);


#if DML_TARGET_VERSION >= 0x6300
        const int antialiased = kernelCreationContext.GetOptionalAttribute<int>(AttrName::Antialiased, 0);
#endif

        // Map ONNX to DML's mode using offsets and rounding direction.
        // These offsets are in addition to the coordinate transform offsets.
        DML_AXIS_DIRECTION roundingDirection = DML_AXIS_DIRECTION_DECREASING;
        if (interpolationMode == DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR)
        {
            std::string nearestMode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::NearestMode, "round_prefer_floor");
            float offsetAdjustment = 0.5f;
            auto optionalNearestModeValue = TryMapStringToIndex(nearestMode, nearestNeighborRoundingModes);
            if (optionalNearestModeValue)
            {
                // The round_prefer_floor mode rounds values to the nearest integer, with half ties rounded toward
                // negative infinity. The increasing rounding direction is correct, albeit unintuitive, because
                // floor(x + 0.5) would return the wrong result, whereas the correct implementation is ceil(x - 0.5).
                // The input offset is positive because positive input offsets translate the output rightward and
                // downward, which (from the perspective of the output) is equivalent to panning the input
                // toward further negative coordinates.
                switch (*optionalNearestModeValue)
                {
                case 0: /*round_prefer_floor*/ roundingDirection = DML_AXIS_DIRECTION_INCREASING; offsetAdjustment =  0.5;  break;
                case 1: /*round_prefer_ceil */ roundingDirection = DML_AXIS_DIRECTION_DECREASING; offsetAdjustment = -0.5;  break;
                case 2: /*floor             */ roundingDirection = DML_AXIS_DIRECTION_DECREASING; offsetAdjustment =  0.0;  break;
                case 3: /*ceil              */ roundingDirection = DML_AXIS_DIRECTION_INCREASING; offsetAdjustment =  0.0;  break;
                default:
                    assert(false);
                }
            }
            if (offsetAdjustment != 0.0f)
            {
                for (auto& offset : inputPixelOffsets)
                {
                    offset += offsetAdjustment;
                }
            }
        }

        // Create the operator description.
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

#if DML_TARGET_VERSION >= 0x6300
        DML_RESAMPLE3_OPERATOR_DESC operatorDesc = {};
        operatorDesc.Antialiased = static_cast<BOOL>(antialiased);
#else
        DML_RESAMPLE2_OPERATOR_DESC operatorDesc = {};
#endif
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.InterpolationMode = interpolationMode;
        operatorDesc.RoundingDirection = roundingDirection;
        operatorDesc.Scales = paddedScales.data();
        operatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(paddedScales.size());
        operatorDesc.InputPixelOffsets = inputPixelOffsets.data();
        operatorDesc.OutputPixelOffsets = outputPixelOffsets.data();
#if DML_TARGET_VERSION >= 0x6300
        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_RESAMPLE3, &operatorDesc };
#else
        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_RESAMPLE2, &operatorDesc };
#endif
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

void CALLBACK QueryResize(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported)
{
    *isSupported = false;

    MLOperatorAttributes attributes(context);

    // DML does not support cubic.
    std::string mode = attributes.GetOptionalAttribute<std::string>(AttrName::Mode, "nearest");
    if (mode == "cubic")
    {
        return;
    }

    // DML clamps the input coordinates to the edges and essentially repeats the last pixel.
    // So rescaling the input kernel total denominator is not supported.
    int32_t excludeOutside = attributes.GetOptionalAttribute<int32_t>(AttrName::ExcludeOutside, 0);
    if (excludeOutside != 0)
    {
        return;
    }

    // DML does not support specifying a specific element value for reading outside the edges.
    // Note the extrapolation value is only pertinent for "tf_crop_and_resize" mode.
    float extrapolationValue = attributes.GetOptionalAttribute<float>(AttrName::ExtrapolationValue, 0.0);
    if (extrapolationValue != 0.0)
    {
        return;
    }

    // Ignore parameter "cubic_coeff_a" since Cubic interpolation unsupported in DML.
    // Ignore parameter "extrapolation_value" as DML clamps to the input rather than reading black pixels.

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(Resize10, VersionedKernel<DmlOperatorResize, 10>);
DML_OP_DEFINE_CREATION_FUNCTION(Resize11, VersionedKernel<DmlOperatorResize, 11>);
DML_OP_DEFINE_CREATION_FUNCTION(Resize13, VersionedKernel<DmlOperatorResize, 13>);
#if DML_TARGET_VERSION >= 0x6300
DML_OP_DEFINE_CREATION_FUNCTION(Resize18, VersionedKernel<DmlOperatorResize, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(Resize19, VersionedKernel<DmlOperatorResize, 19>);
#endif
DML_OP_DEFINE_CREATION_FUNCTION(Upsample7, VersionedKernel<DmlOperatorResize, 7>);
DML_OP_DEFINE_CREATION_FUNCTION(Upsample9, VersionedKernel<DmlOperatorResize, 9>);
DML_OP_DEFINE_CREATION_FUNCTION(Upsample10, VersionedKernel<DmlOperatorResize, 10>);

} // namespace Dml
