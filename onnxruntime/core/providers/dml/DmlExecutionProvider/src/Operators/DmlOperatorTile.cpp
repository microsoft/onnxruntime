// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

        // Because DirectML supports a limited number of dimensions, try to squeeze the dimension count
        // to only those which actually matter. Models sometimes use a greater number of dimensions,
        // even though those dimensions have no significance and can be elided (nop 1's), coercing the
        // total dimension count back down to a supported value.

        std::vector<uint32_t> squeezedInputShape = m_inputDimensions;
        std::vector<uint32_t> squeezedOutputShape = m_outputDimensions;
        std::vector<uint32_t> squeezableDimensionIndices;
        std::vector<uint32_t> paddedRepeatsData = m_repeatsData;
        FindValueIndices<uint32_t>(gsl::make_span(squeezedOutputShape), 1u, /*out*/ squeezableDimensionIndices);

        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ squeezedInputShape);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ paddedRepeatsData);
        RemoveValuesByIndex(squeezableDimensionIndices, /*keepOneValue*/ true, /*inout*/ squeezedOutputShape);

        // Update the tensor descriptions.
        MLOperatorTensorDataType inputTensorDataType = kernelCreationContext.GetInputEdgeDescription(0).tensorDataType;
        auto inputTensorDesc = TensorDesc(inputTensorDataType, squeezedInputShape, squeezedInputShape, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, 1, 0);
        auto outputTensorDesc = TensorDesc(inputTensorDataType, squeezedOutputShape, squeezedOutputShape, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, 1, 0);
        m_inputTensorDescs[0] = inputTensorDesc;
        m_outputTensorDescs[0] = outputTensorDesc;

        // If the output tensor dimension count was right-aligned to a larger size,
        // then ensure that repeat counts have the same count as the tensor rank by
        // inserting leading ones, since DirectML requires them to have the same count.
        const uint32_t squeezedDimCount = gsl::narrow_cast<uint32_t>(squeezedOutputShape.size());
        const uint32_t dmlCompatibleDimCount = outputTensorDesc.GetDimensionCount();
        if (dmlCompatibleDimCount > squeezedDimCount)
        {
            paddedRepeatsData.insert(paddedRepeatsData.begin(), dmlCompatibleDimCount - squeezedDimCount, 1);
        }

        // Create the operator description.
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

DML_OP_DEFINE_CREATION_FUNCTION(Tile, DmlOperatorTile);

} // namespace Dml
