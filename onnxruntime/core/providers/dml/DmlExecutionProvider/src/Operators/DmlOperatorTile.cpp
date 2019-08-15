//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#include "precomp.h"

namespace Dml
{

class DmlOperatorTile : public DmlOperator, TileHelper
{
public:
    using Self = DmlOperatorTile;

    DmlOperatorTile(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        TileHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2, "Tile expects 2 input tensors.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Tile expects 1 output tensor.");

        std::vector<std::optional<uint32_t>> inputIndices = { 0 }; // Use only the first tensor. The second tensor is CPU based and should not be passed to Tile.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);

        // Because DirectML supports a limited number of dimensions, try to reduce the dimension count
        // to only those which actually matter. Models sometimes use a greater number of dimensions,
        // even though those dimensions have no significance and can be elided (nop 1's), coercing the
        // total dimension count back down to a supported value.

        std::vector<uint32_t> squeezedInputShape = m_inputDimensions;
        std::vector<uint32_t> squeezableDimensionIndices;
        std::vector<uint32_t> paddedRepeatsData = m_repeatsData;
        FindValueIndices<uint32_t>(gsl::make_span(m_outputDimensions), 1u, /*out*/ squeezableDimensionIndices);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ squeezedInputShape);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ paddedRepeatsData);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ m_outputDimensions);

        MLOperatorTensorDataType inputTensorDataType = kernelCreationContext.GetInputEdgeDescription(0).tensorDataType;

        TensorDesc inputTensorDesc =
            TensorDesc(
                inputTensorDataType,
                squeezedInputShape,
                squeezedInputShape,
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                NchwDimensionCount, // minDimensionCount
                0
            );

        TensorDesc outputTensorDesc =
            TensorDesc(
                inputTensorDataType,
                gsl::make_span(m_outputDimensions),
                gsl::make_span(m_outputDimensions),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                NchwDimensionCount, // minDimensionCount
                0
            );

        const size_t reducedDimCount = gsl::narrow_cast<uint32_t>(m_outputDimensions.size());
        if (outputTensorDesc.GetDimensionCount() > reducedDimCount)
        {
            paddedRepeatsData.insert(paddedRepeatsData.begin(), outputTensorDesc.GetDimensionCount() - reducedDimCount, 1);
        }

        m_inputTensorDescs[0] = inputTensorDesc;
        m_outputTensorDescs[0] = outputTensorDesc;

        // Create the operator with new shape
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_TILE_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.RepeatsCount = gsl::narrow_cast<uint32_t>(paddedRepeatsData.size());
        operatorDesc.Repeats = paddedRepeatsData.data();

        SetDmlOperatorDesc({ DML_OPERATOR_TILE, &operatorDesc }, kernelCreationContext);
    }
};

// Bug 7325: Reshape and tile are missing graph optimization due to constant CPU tensors
// https://dev.azure.com/microsoft/OS/_workitems/edit/21113503

DML_OP_DEFINE_CREATION_FUNCTION(Tile, DmlOperatorTile);

} // namespace Dml
