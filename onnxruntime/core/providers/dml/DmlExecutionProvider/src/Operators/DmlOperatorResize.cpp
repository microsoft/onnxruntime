// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

constexpr NameAndIndex coordinateTransformationModes[] =
{
    {"half_pixel", 0},
    {"pytorch_half_pixel", 1},
    {"align_corners", 2},
    {"asymmetric", 3},
    {"tf_half_pixel_for_nn", 4},
    {"tf_crop_and_resize", 5},
};

constexpr NameAndIndex nearestNeighborRoundingModes[] =
{
    {"", 0},
    {"round_prefer_floor", 0},
    {"round_prefer_ceil", 1},
    {"floor", 2},
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
        !regionOfInterest.empty() || coordinateTransformationModeValue != 5 /*tf_crop_and_resize*/,
        "Resize expects 'roi' tensor for 'tf_crop_and_resize' mode."
    );

    const uint32_t rank = gsl::narrow_cast<uint32_t>(inputDimensions.size());

    // Fill in all the input/output pixel offset for each axis,
    // and recompute the scale for certain modes.

    for (uint32_t i = 0; i < rank; ++i)
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

        case 2:
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

        case 3:
            // if coordinate_transformation_mode is "asymmetric",
            // x_original = x_resized / scale
            inputPixelOffset = 0.0;
            outputPixelOffset = 0.0;
            // Keep existing scales.
            break;

        case 4:
            // if coordinate_transformation_mode is "tf_half_pixel_for_nn",
            // x_original = (x_resized + 0.5) / scale
            inputPixelOffset = 0.0;
            outputPixelOffset = -0.5;
            // Keep existing scales.
            break;

        case 5:
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
        for (size_t i = 0, rank = m_outputDimensions.size(); i < rank; ++i)
        {
            if (m_inputDimensions[i] = 1 && m_outputDimensions[i] == 1)
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

        // DML's nearest neighbor mode uses round-halves-up (or round_prefer_ceil) via floor(input.x + 0.5).
        // So to support floor, adjust the input by half a pixel.
        // round_prefer_floor is not supported without an API extension,
        // but existing code already default to treating it as round_prefer_ceil.
        // So continue that.
        if (interpolationMode == DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR)
        {
            std::string nearestMode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::NearestMode, "round_prefer_floor");
            auto optionalNearestModeValue = TryMapStringToIndex(nearestMode, nearestNeighborRoundingModes);
            if (optionalNearestModeValue)
            {
                switch (*optionalNearestModeValue)
                {
                case 0: // round_prefer_floor
                case 1: // round_prefer_ceil
                    break;
                case 2: // floor
                    for (auto& offset : inputPixelOffsets)
                    {
                        offset += 0.5;
                    }
                    break;
                default:
                    assert(false);
                }
            }
        }

        // Create the operator description.
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_RESAMPLE1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.InterpolationMode = interpolationMode;
        operatorDesc.Scales = paddedScales.data();
        operatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(paddedScales.size());
        operatorDesc.InputPixelOffsets = inputPixelOffsets.data();
        operatorDesc.OutputPixelOffsets = outputPixelOffsets.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_RESAMPLE1, &operatorDesc };
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

    // DML's nearest neighbor mode uses half pixels rounded down.
    std::string nearestMode = attributes.GetOptionalAttribute<std::string>(AttrName::NearestMode, "round_prefer_floor");
    auto optionalNearestModeValue = TryMapStringToIndex(nearestMode, nearestNeighborRoundingModes);
    if (!optionalNearestModeValue)
    {
        return;
    }

    // Ignore parameter "cubic_coeff_a" since Cubic interpolation unsupported in DML.
    // Ignore parameter "extrapolation_value" as DML clamps to the input rather than reading black pixels.

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(Resize10, VersionedKernel<DmlOperatorResize, 10>);
DML_OP_DEFINE_CREATION_FUNCTION(Resize11, VersionedKernel<DmlOperatorResize, 11>);
DML_OP_DEFINE_CREATION_FUNCTION(Upsample7, VersionedKernel<DmlOperatorResize, 7>);
DML_OP_DEFINE_CREATION_FUNCTION(Upsample9, VersionedKernel<DmlOperatorResize, 9>);
DML_OP_DEFINE_CREATION_FUNCTION(Upsample10, VersionedKernel<DmlOperatorResize, 10>);

} // namespace Dml
