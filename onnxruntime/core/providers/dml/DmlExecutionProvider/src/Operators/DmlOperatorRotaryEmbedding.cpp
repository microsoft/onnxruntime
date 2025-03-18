// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

// This operator is easier to understand by looking at a python implementation of the non-interleaved version:
//
// def rotate_half(x):
//     """Rotates half the hidden dims of the input."""
//     half_dim = x.shape[-1] // 2
//     x1 = x[..., :half_dim]
//     x2 = x[..., half_dim:]
//     return np.concatenate((-x2, x1), dim=-1)
//
//
// def apply_rope(x, cos, sin, position_ids):
//     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
//     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
//     x_embed = (x * cos) + (rotate_half(x) * sin)
//     return x_embed
//
// For the non-interleaved version, we multiply the cos cache by the non-rotated input tensor while we multiply the sin cache
// by the rotated input tensor. Rotating the tensor means slicing it in half on the head dimension and swapping the 2 halves.
//
// The interleaved version is very similar but instead of swapping 2 halves, we swap every pair of adjacent elements and we swap
// the sign of every adjacent element.

// Here's a representation of what the graph looks like in DML, before getting fused together:
/*
                 Input                            CosCache   PositionIds     SinCache
                   |                                 |           |              |
                   |                                 |  +--------+-----------+  |
                 Split                               |  |                    |  |
                  |  |                              Gather                  Gather
          +-------+  |                                |                        |
          |          |                                |                        |
          |     Identity----------+                   |                        |
          |        |              |                   |                        |
          |        |              |                   |                        |
          |    --Split--          |                   |                        |
          |    \       /          | +-----------------+                        |
          |     \     /           | |                                          |
          |      \   /            Mul                                          |
          |       \ /              |                                           |
          |        X               |                                           |
          |       / \              |                                           |
          |      /   \             |                                           |
          |       Join             |                                           |
          |        |               |                                           |
          |        | +---------------------------------------------------------+
          |        | |             |
          |        Mul             |
          |         |              |
          |         +-----+ +------+
          |               | |
          |               Add
          |                |
          +-------------+  |
                        |  |
                        Join
*/

namespace Dml
{
class DmlOperatorRotaryEmbedding : public DmlOperator
{
public:
    DmlOperatorRotaryEmbedding(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        enum InputIndex : uint32_t
        {
            inputDataIndex,
            positionIdsIndex,
            cosCacheIndex,
            sinCacheIndex,
        };

        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 4);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        // When the input is 4D, it has the shape [batchSize, numHeads, sequenceLength, headSize]. Otherwise,
        // it has the shape [batchSize, sequenceLength, hiddenSize]
        const bool inputIs4D = kernelInfo.GetInputTensorDimensionCount(inputDataIndex) == 4;

        // When positionIds is a scalar, it represents the start offset for each sequence
        const bool positionIdsIsOffset = kernelInfo.GetInputTensorDimensionCount(positionIdsIndex) == 1;

        Initialize(kernelInfo);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[inputDataIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[positionIdsIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[cosCacheIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[sinCacheIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs[0].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[cosCacheIndex].GetSizes() == m_inputTensorDescs[sinCacheIndex].GetSizes());

        uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelInfo.GetOptionalAttribute<int64_t>(AttrName::NumHeads, 0));
        uint32_t rotaryEmbeddingDim = gsl::narrow_cast<uint32_t>(kernelInfo.GetOptionalAttribute<int64_t>(AttrName::RotaryEmbeddingDim, 0));

        const auto inputDataSizes = m_inputTensorDescs[inputDataIndex].GetSizes();
        const uint32_t hiddenSize = inputIs4D ? inputDataSizes[1] * inputDataSizes[3] : inputDataSizes.back();

        const uint32_t headSize = numHeads == 0
            ? m_inputTensorDescs[cosCacheIndex].GetSizes().back() * 2
            : hiddenSize / numHeads;

        if (rotaryEmbeddingDim > 0)
        {
            ORT_ENFORCE(numHeads > 0, "num_heads must be provided if rotary_embedding_dim is specified");
        }
        else
        {
            rotaryEmbeddingDim = headSize;
        }

        if (numHeads == 0)
        {
            numHeads = hiddenSize / headSize;
        }
        else if (inputIs4D)
        {
            ORT_ENFORCE(numHeads == inputDataSizes[1], "When the input has 4 dimensions, num_heads must be 0 or have the same value as the second dimension of the input");
        }

        const uint32_t batchSize = inputIs4D ? inputDataSizes[0] : inputDataSizes[1];
        const uint32_t sequenceLength = inputDataSizes[2];

        const auto cosCacheSizes = m_inputTensorDescs[cosCacheIndex].GetSizes();
        const uint32_t maxSequenceLength = cosCacheSizes[cosCacheSizes.size() - 2];

        const bool isPackedBatching = gsl::narrow_cast<uint32_t>(kernelInfo.GetOptionalAttribute<int64_t>(AttrName::IsPackedBatching, 0)) == 1;
        if (!isPackedBatching && sequenceLength > maxSequenceLength)
        {
            ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
        }

        const bool interleaved = gsl::narrow_cast<bool>(kernelInfo.GetOptionalAttribute<int64_t>(AttrName::Interleaved, 0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        const MLOperatorTensorDataType dataType = kernelInfo.GetInputEdgeDescription(inputDataIndex).tensorDataType;

         // We resize the data to be of shape [batchSize, sequenceLength, numHeads, headSize]
        const std::array<uint32_t, 4> inputOutputShape = inputIs4D
            ? std::array<uint32_t, 4>({batchSize, numHeads, sequenceLength, headSize})
            : std::array<uint32_t, 4>({batchSize, sequenceLength, numHeads, headSize});

        const std::array<uint32_t, 4> splitInputOutputShape1 = inputIs4D
            ? std::array<uint32_t, 4>({batchSize, numHeads, sequenceLength, rotaryEmbeddingDim})
            : std::array<uint32_t, 4>({batchSize, sequenceLength, numHeads, rotaryEmbeddingDim});

        const std::array<uint32_t, 4> splitInputOutputShape2 = inputIs4D
            ? std::array<uint32_t, 4>({batchSize, numHeads, sequenceLength, headSize - rotaryEmbeddingDim})
            : std::array<uint32_t, 4>({batchSize, sequenceLength, numHeads, headSize - rotaryEmbeddingDim});

        TensorDesc inputOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, inputOutputShape);
        TensorDesc splitInputOutputTensorDesc1 = TensorDesc::ConstructDefaultTensorDesc(dataType, splitInputOutputShape1);
        TensorDesc splitInputOutputTensorDesc2 = TensorDesc::ConstructDefaultTensorDesc(dataType, splitInputOutputShape2);

        // Split the input to perform the rotary embedding only on a subregion of the tensor if needed. The split inputs
        // will be joined back together at the end.
        const DML_TENSOR_DESC inputOutputDmlTensorDesc = inputOutputTensorDesc.GetDmlDesc();

        std::array<DML_TENSOR_DESC, 2> splitTensorDescs = {
            splitInputOutputTensorDesc1.GetDmlDesc(),
            splitInputOutputTensorDesc2.GetDmlDesc(),
        };

        DML_SPLIT_OPERATOR_DESC splitInputOperatorDesc{};
        DML_OPERATOR_DESC splitInputDmlOperatorDesc{};
        if (headSize != rotaryEmbeddingDim)
        {
            splitInputOperatorDesc.InputTensor = &inputOutputDmlTensorDesc;
            splitInputOperatorDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitTensorDescs.size());
            splitInputOperatorDesc.OutputTensors = splitTensorDescs.data();
            splitInputOperatorDesc.Axis = gsl::narrow_cast<uint32_t>(inputOutputShape.size()) - 1;
            splitInputDmlOperatorDesc.Type = DML_OPERATOR_SPLIT;
            splitInputDmlOperatorDesc.Desc = &splitInputOperatorDesc;
        }

        // Copy the partial input to preserve its real input shape in the graph without reshaping it. This will disappear during DML's graph compilation phase.
        DML_SCALE_BIAS scaleBias = {1.0f, 0.0f};

        const std::array<uint32_t, 4> partialInputOutputShape = {batchSize, sequenceLength, numHeads, rotaryEmbeddingDim};
        TensorDesc partialStridedInputOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, partialInputOutputShape);
        TensorDesc partialInputOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, partialInputOutputShape);

        if (inputIs4D)
        {
            const std::array<uint32_t, 4> partialInputOutputStrides = {rotaryEmbeddingDim * numHeads * sequenceLength, rotaryEmbeddingDim, sequenceLength * rotaryEmbeddingDim, 1};
            partialStridedInputOutputTensorDesc.SetStrides(partialInputOutputStrides);
        }

        const DML_TENSOR_DESC partialStridedInputOutputDmlTensorDesc = partialStridedInputOutputTensorDesc.GetDmlDesc();
        const DML_TENSOR_DESC partialInputOutputDmlTensorDesc = partialInputOutputTensorDesc.GetDmlDesc();

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC copyInputDesc{};
        copyInputDesc.InputTensor = &partialStridedInputOutputDmlTensorDesc;
        copyInputDesc.OutputTensor = &partialInputOutputDmlTensorDesc;
        copyInputDesc.ScaleBias = &scaleBias;
        const DML_OPERATOR_DESC copyInputDmlDesc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY, &copyInputDesc};

        const uint32_t halfRoraryEmbeddingDim = rotaryEmbeddingDim / 2;

        // Split the input data into 2 equal parts
        const std::vector<uint32_t> partialInputDataTensorShape = interleaved
            ? std::vector<uint32_t>({batchSize, sequenceLength, numHeads, rotaryEmbeddingDim / 2, 2})
            : std::vector<uint32_t>({batchSize, sequenceLength, numHeads, 2, rotaryEmbeddingDim / 2});

        const std::vector<uint32_t> splitInputDataTensorShape = interleaved
            ? std::vector<uint32_t>({batchSize, sequenceLength, numHeads, rotaryEmbeddingDim / 2, 1})
            : std::vector<uint32_t>({batchSize, sequenceLength, numHeads, 1, rotaryEmbeddingDim / 2});

        TensorDesc partialInputDataTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, partialInputDataTensorShape);
        const DML_TENSOR_DESC partialInputDataDmlTensorDesc = partialInputDataTensorDesc.GetDmlDesc();

        TensorDesc joinedDataTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, partialInputDataTensorShape);
        const DML_TENSOR_DESC joinedDataDmlTensorDesc = joinedDataTensorDesc.GetDmlDesc();

        TensorDesc splitInputDataTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, splitInputDataTensorShape);
        const std::array<DML_TENSOR_DESC, 2> splitInputDataDmlTensorDescs = {splitInputDataTensorDesc.GetDmlDesc(), splitInputDataTensorDesc.GetDmlDesc()};

        DML_SPLIT_OPERATOR_DESC splitPartialInputDesc{};
        splitPartialInputDesc.InputTensor = &partialInputDataDmlTensorDesc;
        splitPartialInputDesc.OutputTensors = splitInputDataDmlTensorDescs.data();
        splitPartialInputDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitInputDataDmlTensorDescs.size());
        splitPartialInputDesc.Axis = interleaved
            ? gsl::narrow_cast<uint32_t>(splitInputDataTensorShape.size()) - 1
            : gsl::narrow_cast<uint32_t>(splitInputDataTensorShape.size()) - 2;

        const DML_OPERATOR_DESC splitPartialInputDmlDesc = {DML_OPERATOR_SPLIT, &splitPartialInputDesc};

        // Swap the 2 halves and join them together
        DML_JOIN_OPERATOR_DESC joinPartialInputDesc{};
        joinPartialInputDesc.InputTensors = splitInputDataDmlTensorDescs.data();
        joinPartialInputDesc.OutputTensor = &joinedDataDmlTensorDesc;
        joinPartialInputDesc.Axis = splitPartialInputDesc.Axis;
        joinPartialInputDesc.InputCount = gsl::narrow_cast<uint32_t>(splitInputDataDmlTensorDescs.size());
        const DML_OPERATOR_DESC joinPartialInputDmlDesc = {DML_OPERATOR_JOIN, &joinPartialInputDesc};

        // We generate a sequence from 0 to sequenceLength and add the offset to it
        const std::array<uint32_t, 4> positionIdsRangeShape = {1, 1, 1, sequenceLength};
        auto positionIdsDataType = kernelInfo.GetInputEdgeDescription(positionIdsIndex).tensorDataType;
        TensorDesc positionIdsRangeTensorDesc = TensorDesc::ConstructDefaultTensorDesc(positionIdsDataType, positionIdsRangeShape);
        const DML_TENSOR_DESC positionIdsRangeDmlTensorDesc = positionIdsRangeTensorDesc.GetDmlDesc();

        const std::array<uint32_t, 4> broadcastedPositionIdsRangeShape = {1, 1, batchSize, sequenceLength};
        TensorDesc broadcastedPositionIdsRangeTensorDesc = TensorDesc::ConstructBroadcastedTensorDesc(positionIdsDataType, broadcastedPositionIdsRangeShape, positionIdsRangeShape);
        const DML_TENSOR_DESC broadcastedPositionIdsRangeDmlTensorDesc = broadcastedPositionIdsRangeTensorDesc.GetDmlDesc();

        const std::array<uint32_t, 4> broadcastedOffsetShape = {1, 1, batchSize, sequenceLength};
        TensorDesc broadcastedOffsetTensorDesc = TensorDesc::ConstructBroadcastedTensorDesc(positionIdsDataType, broadcastedOffsetShape, m_inputTensorDescs[positionIdsIndex].GetSizes());
        const DML_TENSOR_DESC broadcastedOffsetDmlTensorDesc = broadcastedOffsetTensorDesc.GetDmlDesc();

        TensorDesc offsetPositionIdsTensorDesc = TensorDesc::ConstructDefaultTensorDesc(positionIdsDataType, broadcastedOffsetShape);
        const DML_TENSOR_DESC offsetPositionIdsRangeDmlTensorDesc = offsetPositionIdsTensorDesc.GetDmlDesc();

        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC positionIdsRange{};
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC positionIdsAddOffset{};
        if (positionIdsIsOffset)
        {
            ML_CHECK_VALID_ARGUMENT(positionIdsDataType == MLOperatorTensorDataType::Int64);
            positionIdsRange.ValueDataType = DML_TENSOR_DATA_TYPE_INT64;
            positionIdsRange.ValueDelta.Int64 = 1;
            positionIdsRange.OutputTensor = &positionIdsRangeDmlTensorDesc;

            positionIdsAddOffset.ATensor = &broadcastedPositionIdsRangeDmlTensorDesc;
            positionIdsAddOffset.BTensor = &broadcastedOffsetDmlTensorDesc;
            positionIdsAddOffset.OutputTensor = &offsetPositionIdsRangeDmlTensorDesc;
        }
        const DML_OPERATOR_DESC positionIdsRangeDmlDesc = {DML_OPERATOR_FILL_VALUE_SEQUENCE, &positionIdsRange};
        const DML_OPERATOR_DESC positionIdsAddOffsetDmlDesc = {DML_OPERATOR_ELEMENT_WISE_ADD, &positionIdsAddOffset};

        // Gather the cos/sin values based on the position ids
        const std::array<uint32_t, 4> gatheredCosSinShape = {1, batchSize, sequenceLength, rotaryEmbeddingDim / 2};
        TensorDesc gatheredCosSinTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, gatheredCosSinShape);
        const DML_TENSOR_DESC gatheredCosSinDmlTensorDesc = gatheredCosSinTensorDesc.GetDmlDesc();

        DML_GATHER_OPERATOR_DESC gatherCosSinDesc{};
        gatherCosSinDesc.InputTensor = &inputDescs[cosCacheIndex];
        gatherCosSinDesc.IndicesTensor = positionIdsIsOffset ? &offsetPositionIdsRangeDmlTensorDesc : &inputDescs[positionIdsIndex];
        gatherCosSinDesc.OutputTensor = &gatheredCosSinDmlTensorDesc;
        gatherCosSinDesc.Axis = 2;
        gatherCosSinDesc.IndexDimensions = 2;
        const DML_OPERATOR_DESC gatherCosSinDmlDesc {DML_OPERATOR_GATHER, &gatherCosSinDesc};

        // After gathering cos/sin, reshape and broadcast them to match the number of heads of the input data
        const std::vector<uint32_t> reshapedCosSinShape = interleaved
            ? std::vector<uint32_t>({batchSize, sequenceLength, 1, rotaryEmbeddingDim / 2, 1})
            : std::vector<uint32_t>({batchSize, sequenceLength, 1, 1, rotaryEmbeddingDim / 2});
        TensorDesc broadcastedCosSinTensorDesc = TensorDesc::ConstructBroadcastedTensorDesc(dataType, partialInputDataTensorShape, reshapedCosSinShape);
        const DML_TENSOR_DESC broadcastedCosSinDmlTensorDesc = broadcastedCosSinTensorDesc.GetDmlDesc();

        // Create a vector that contains the sign values {-1, 1}
        const std::array<uint32_t, 1> signTensorShape = {2};
        TensorDesc signTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, signTensorShape);
        const DML_TENSOR_DESC signDmlTensorDesc = signTensorDesc.GetDmlDesc();

        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC signRange{};
        signRange.OutputTensor = &signDmlTensorDesc;
        if (dataType == MLOperatorTensorDataType::Float16)
        {
            const auto valueStart = static_cast<MLFloat16>(-1.0f);
            const auto valueDelta = static_cast<MLFloat16>(2.0f);
            memcpy(signRange.ValueStart.Bytes, reinterpret_cast<const BYTE*>(&valueStart), sizeof(valueStart));
            memcpy(signRange.ValueDelta.Bytes, reinterpret_cast<const BYTE*>(&valueDelta), sizeof(valueDelta));
            signRange.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(dataType == MLOperatorTensorDataType::Float);
            signRange.ValueStart.Float32 = -1.0f;
            signRange.ValueDelta.Float32 = 2.0f;
            signRange.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
        }
        const DML_OPERATOR_DESC signRangeDmlDesc = {DML_OPERATOR_FILL_VALUE_SEQUENCE, &signRange};

        // Multiply the broadcasted sign values with the rotated input
        const std::vector<uint32_t> reshapedSignShape = interleaved
            ? std::vector<uint32_t>({1, 1, 1, 1, 2})
            : std::vector<uint32_t>({1, 1, 1, 2, 1});
        TensorDesc broadcastedSignCosSinTensorDesc = TensorDesc::ConstructBroadcastedTensorDesc(dataType, partialInputDataTensorShape, reshapedSignShape);
        const DML_TENSOR_DESC broadcastedSignDmlTensorDesc = broadcastedSignCosSinTensorDesc.GetDmlDesc();

        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulSignDesc{};
        mulSignDesc.ATensor = &joinedDataDmlTensorDesc;
        mulSignDesc.BTensor = &broadcastedSignDmlTensorDesc;
        mulSignDesc.OutputTensor = &joinedDataDmlTensorDesc;
        const DML_OPERATOR_DESC mulSignDmlDesc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulSignDesc};

        // Multiply the non-rotated data with the cos and the rotated data with the sin
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulCosSinDesc{};
        mulCosSinDesc.ATensor = &joinedDataDmlTensorDesc;
        mulCosSinDesc.BTensor = &broadcastedCosSinDmlTensorDesc;
        mulCosSinDesc.OutputTensor = &joinedDataDmlTensorDesc;
        const DML_OPERATOR_DESC mulCosSinDmlDesc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulCosSinDesc};

        // Add the multiplied cos and sin values together
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc{};
        addDesc.ATensor = &partialInputOutputDmlTensorDesc;
        addDesc.BTensor = &partialInputOutputDmlTensorDesc;
        addDesc.OutputTensor = &partialStridedInputOutputDmlTensorDesc;
        const DML_OPERATOR_DESC addDmlDesc = {DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc};

        DML_JOIN_OPERATOR_DESC joinOutputOperatorDesc{};
        DML_OPERATOR_DESC joinOutputDmlOperatorDesc{};
        if (headSize != rotaryEmbeddingDim)
        {
            joinOutputOperatorDesc.InputCount = gsl::narrow_cast<uint32_t>(splitTensorDescs.size());
            joinOutputOperatorDesc.InputTensors = splitTensorDescs.data();
            joinOutputOperatorDesc.OutputTensor = &inputOutputDmlTensorDesc;
            joinOutputOperatorDesc.Axis = gsl::narrow_cast<uint32_t>(inputOutputShape.size()) - 1;
            joinOutputDmlOperatorDesc.Type = DML_OPERATOR_JOIN;
            joinOutputDmlOperatorDesc.Desc = &joinOutputOperatorDesc;
        }

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        std::vector<const DML_OPERATOR_DESC*> opDescs = {
            &copyInputDmlDesc, // Copy the input data to preseve the real input shape
            &splitPartialInputDmlDesc, // Split the input data
            &gatherCosSinDmlDesc, // Gather cos
            &gatherCosSinDmlDesc, // Gather sin
            &signRangeDmlDesc, // Generate the signs

            &joinPartialInputDmlDesc, // Join the split data
            &mulCosSinDmlDesc, // Multiply cos with the non-rotated data
            &mulCosSinDmlDesc, // Multiply sin with the rotated data
            &mulSignDmlDesc, // Multiply the sign with the rotated data
            &addDmlDesc, // Add the rotated cos and non-rotated sin parts together
        };

        enum NodeIndex : uint32_t
        {
            copyInputOpIndex,
            splitPartialInputOpIndex,
            gatherCosOpIndex,
            gatherSinOpIndex,
            signRangeOpIndex,

            joinPartialInputOpIndex,
            mulCosOpIndex,
            mulSinOpIndex,
            mulSignOpIndex,
            addOpIndex,

            // The following indices are optional
            positionIdsRangeOpIndex,
            positionIdsAddOffsetOpIndex,
        };

        uint32_t splitInputOpIndex = positionIdsIsOffset ? positionIdsAddOffsetOpIndex + 1 : addOpIndex + 1;
        uint32_t joinOutputOpIndex = splitInputOpIndex + 1;

        if (positionIdsIsOffset)
        {
            opDescs.push_back(&positionIdsRangeDmlDesc);
            opDescs.push_back(&positionIdsAddOffsetDmlDesc);

            DML_INPUT_GRAPH_EDGE_DESC positionIdsToAddOffsetEdge = {};
            positionIdsToAddOffsetEdge.GraphInputIndex = positionIdsIndex;
            positionIdsToAddOffsetEdge.ToNodeIndex = positionIdsAddOffsetOpIndex;
            positionIdsToAddOffsetEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(positionIdsToAddOffsetEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC positionIdsOffsetToAddOffsetEdge = {};
            positionIdsOffsetToAddOffsetEdge.FromNodeIndex = positionIdsRangeOpIndex;
            positionIdsOffsetToAddOffsetEdge.FromNodeOutputIndex = 0;
            positionIdsOffsetToAddOffsetEdge.ToNodeIndex = positionIdsAddOffsetOpIndex;
            positionIdsOffsetToAddOffsetEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(positionIdsOffsetToAddOffsetEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC positionIdsAddOffsetToGatherCosEdge = {};
            positionIdsAddOffsetToGatherCosEdge.FromNodeIndex = positionIdsAddOffsetOpIndex;
            positionIdsAddOffsetToGatherCosEdge.FromNodeOutputIndex = 0;
            positionIdsAddOffsetToGatherCosEdge.ToNodeIndex = gatherCosOpIndex;
            positionIdsAddOffsetToGatherCosEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(positionIdsAddOffsetToGatherCosEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC positionIdsAddOffsetToGatherSinEdge = {};
            positionIdsAddOffsetToGatherSinEdge.FromNodeIndex = positionIdsAddOffsetOpIndex;
            positionIdsAddOffsetToGatherSinEdge.FromNodeOutputIndex = 0;
            positionIdsAddOffsetToGatherSinEdge.ToNodeIndex = gatherSinOpIndex;
            positionIdsAddOffsetToGatherSinEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(positionIdsAddOffsetToGatherSinEdge);
        }
        else
        {
            DML_INPUT_GRAPH_EDGE_DESC positionIdsToGatherCosEdge = {};
            positionIdsToGatherCosEdge.GraphInputIndex = positionIdsIndex;
            positionIdsToGatherCosEdge.ToNodeIndex = gatherCosOpIndex;
            positionIdsToGatherCosEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(positionIdsToGatherCosEdge);

            DML_INPUT_GRAPH_EDGE_DESC positionIdsToGatherSinEdge = {};
            positionIdsToGatherSinEdge.GraphInputIndex = positionIdsIndex;
            positionIdsToGatherSinEdge.ToNodeIndex = gatherSinOpIndex;
            positionIdsToGatherSinEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(positionIdsToGatherSinEdge);
        }

        if (splitInputDmlOperatorDesc.Desc)
        {
            opDescs.push_back(&splitInputDmlOperatorDesc);
            opDescs.push_back(&joinOutputDmlOperatorDesc);

            DML_INPUT_GRAPH_EDGE_DESC inputToSplitInputEdge = {};
            inputToSplitInputEdge.GraphInputIndex = inputDataIndex;
            inputToSplitInputEdge.ToNodeIndex = splitInputOpIndex;
            inputToSplitInputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(inputToSplitInputEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC partialInputToCopyInputEdge = {};
            partialInputToCopyInputEdge.FromNodeIndex = splitInputOpIndex;
            partialInputToCopyInputEdge.FromNodeOutputIndex = 0;
            partialInputToCopyInputEdge.ToNodeIndex = copyInputOpIndex;
            partialInputToCopyInputEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(partialInputToCopyInputEdge);
        }
        else
        {
            DML_INPUT_GRAPH_EDGE_DESC inputToCopyInputEdge = {};
            inputToCopyInputEdge.GraphInputIndex = inputDataIndex;
            inputToCopyInputEdge.ToNodeIndex = copyInputOpIndex;
            inputToCopyInputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(inputToCopyInputEdge);
        }

        DML_INPUT_GRAPH_EDGE_DESC cosToGatherEdge = {};
        cosToGatherEdge.GraphInputIndex = cosCacheIndex;
        cosToGatherEdge.ToNodeIndex = gatherCosOpIndex;
        cosToGatherEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(cosToGatherEdge);

        DML_INPUT_GRAPH_EDGE_DESC sinToGatherEdge = {};
        sinToGatherEdge.GraphInputIndex = sinCacheIndex;
        sinToGatherEdge.ToNodeIndex = gatherSinOpIndex;
        sinToGatherEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(sinToGatherEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC inputToSplitEdge = {};
        inputToSplitEdge.FromNodeIndex = copyInputOpIndex;
        inputToSplitEdge.FromNodeOutputIndex = 0;
        inputToSplitEdge.ToNodeIndex = splitPartialInputOpIndex;
        inputToSplitEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(inputToSplitEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC nonRotatedDataToMulEdge = {};
        nonRotatedDataToMulEdge.FromNodeIndex = copyInputOpIndex;
        nonRotatedDataToMulEdge.FromNodeOutputIndex = 0;
        nonRotatedDataToMulEdge.ToNodeIndex = mulCosOpIndex;
        nonRotatedDataToMulEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(nonRotatedDataToMulEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC secondHalfDataToJoinEdge = {};
        secondHalfDataToJoinEdge.FromNodeIndex = splitPartialInputOpIndex;
        secondHalfDataToJoinEdge.FromNodeOutputIndex = 1;
        secondHalfDataToJoinEdge.ToNodeIndex = joinPartialInputOpIndex;
        secondHalfDataToJoinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(secondHalfDataToJoinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC firstHalfDataToJoinEdge = {};
        firstHalfDataToJoinEdge.FromNodeIndex = splitPartialInputOpIndex;
        firstHalfDataToJoinEdge.FromNodeOutputIndex = 0;
        firstHalfDataToJoinEdge.ToNodeIndex = joinPartialInputOpIndex;
        firstHalfDataToJoinEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(firstHalfDataToJoinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC cosToMulEdge = {};
        cosToMulEdge.FromNodeIndex = gatherCosOpIndex;
        cosToMulEdge.FromNodeOutputIndex = 0;
        cosToMulEdge.ToNodeIndex = mulCosOpIndex;
        cosToMulEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(cosToMulEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC rotatedDataToMulEdge = {};
        rotatedDataToMulEdge.FromNodeIndex = joinPartialInputOpIndex;
        rotatedDataToMulEdge.FromNodeOutputIndex = 0;
        rotatedDataToMulEdge.ToNodeIndex = mulSinOpIndex;
        rotatedDataToMulEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(rotatedDataToMulEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC sinToMulEdge = {};
        sinToMulEdge.FromNodeIndex = gatherSinOpIndex;
        sinToMulEdge.FromNodeOutputIndex = 0;
        sinToMulEdge.ToNodeIndex = mulSinOpIndex;
        sinToMulEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(sinToMulEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC rotatedSinToMulEdge = {};
        rotatedSinToMulEdge.FromNodeIndex = mulSinOpIndex;
        rotatedSinToMulEdge.FromNodeOutputIndex = 0;
        rotatedSinToMulEdge.ToNodeIndex = mulSignOpIndex;
        rotatedSinToMulEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(rotatedSinToMulEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC signToMulEdge = {};
        signToMulEdge.FromNodeIndex = signRangeOpIndex;
        signToMulEdge.FromNodeOutputIndex = 0;
        signToMulEdge.ToNodeIndex = mulSignOpIndex;
        signToMulEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(signToMulEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC nonRotatedCosToAddEdge = {};
        nonRotatedCosToAddEdge.FromNodeIndex = mulCosOpIndex;
        nonRotatedCosToAddEdge.FromNodeOutputIndex = 0;
        nonRotatedCosToAddEdge.ToNodeIndex = addOpIndex;
        nonRotatedCosToAddEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(nonRotatedCosToAddEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC rotatedSinToAddEdge = {};
        rotatedSinToAddEdge.FromNodeIndex = mulSignOpIndex;
        rotatedSinToAddEdge.FromNodeOutputIndex = 0;
        rotatedSinToAddEdge.ToNodeIndex = addOpIndex;
        rotatedSinToAddEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(rotatedSinToAddEdge);

        if (splitInputDmlOperatorDesc.Desc)
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC addToJoinOutputEdge = {};
            addToJoinOutputEdge.FromNodeIndex = addOpIndex;
            addToJoinOutputEdge.FromNodeOutputIndex = 0;
            addToJoinOutputEdge.ToNodeIndex = joinOutputOpIndex;
            addToJoinOutputEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(addToJoinOutputEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC remainingInputToJoinOutputEdge = {};
            remainingInputToJoinOutputEdge.FromNodeIndex = splitInputOpIndex;
            remainingInputToJoinOutputEdge.FromNodeOutputIndex = 1;
            remainingInputToJoinOutputEdge.ToNodeIndex = joinOutputOpIndex;
            remainingInputToJoinOutputEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(remainingInputToJoinOutputEdge);

            DML_OUTPUT_GRAPH_EDGE_DESC joinOutputToOutputEdge = {};
            joinOutputToOutputEdge.FromNodeIndex = joinOutputOpIndex;
            joinOutputToOutputEdge.FromNodeOutputIndex = 0;
            joinOutputToOutputEdge.GraphOutputIndex = 0;
            outputEdges.push_back(joinOutputToOutputEdge);
        }
        else
        {
            DML_OUTPUT_GRAPH_EDGE_DESC addToOutputEdge = {};
            addToOutputEdge.FromNodeIndex = addOpIndex;
            addToOutputEdge.FromNodeOutputIndex = 0;
            addToOutputEdge.GraphOutputIndex = 0;
            outputEdges.push_back(addToOutputEdge);
        }

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();
        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();
        operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodes = opDescs.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(RotaryEmbedding, DmlOperatorRotaryEmbedding);

} // namespace Dml
