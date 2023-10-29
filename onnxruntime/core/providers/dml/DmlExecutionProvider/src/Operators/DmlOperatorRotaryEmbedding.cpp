// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

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

        Initialize(kernelInfo);

        ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[inputDataIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[positionIdsIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[cosCacheIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[sinCacheIndex].GetDimensionCount() == 4);

        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs[0].GetDimensionCount() == 4);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[cosCacheIndex].GetSizes() == m_inputTensorDescs[sinCacheIndex].GetSizes());
        const uint32_t headSize = m_inputTensorDescs[cosCacheIndex].GetSizes().back() * 2;

        // The last dimension of the data is the hidden size, so it must be divisible by the head size
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[inputDataIndex].GetSizes().back() % headSize == 0);

        // We resize the data to be of shape [batchSize, sequenceLength, numHeads, headSize]
        const auto inputDataSizes = m_inputTensorDescs[inputDataIndex].GetSizes();
        const uint32_t batchSize = inputDataSizes[1];
        const uint32_t sequenceLength = inputDataSizes[2];
        const uint32_t numHeads = inputDataSizes[3] / headSize;

        // Split the input data into 2 equal parts
        const MLOperatorTensorDataType dataType = kernelInfo.GetInputEdgeDescription(inputDataIndex).tensorDataType;
        const std::array<uint32_t, 4> inputDataTensorShape {batchSize, sequenceLength, numHeads, headSize};
        TensorDesc inputDataTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, inputDataTensorShape);
        const DML_TENSOR_DESC inputDataDmlTensorDesc = inputDataTensorDesc.GetDmlDesc();

        const std::array<uint32_t, 4> splitInputDataTensorShape {batchSize, sequenceLength, numHeads, headSize / 2};
        TensorDesc splitInputDataTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, splitInputDataTensorShape);
        const std::array<DML_TENSOR_DESC, 2> splitInputDataDmlTensorDescs = {splitInputDataTensorDesc.GetDmlDesc(), splitInputDataTensorDesc.GetDmlDesc()};

        DML_SPLIT_OPERATOR_DESC splitInputDesc{};
        splitInputDesc.InputTensor = &inputDataDmlTensorDesc;
        splitInputDesc.OutputTensors = splitInputDataDmlTensorDescs.data();
        splitInputDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitInputDataDmlTensorDescs.size());
        splitInputDesc.Axis = gsl::narrow_cast<uint32_t>(splitInputDataTensorShape.size()) - 1;
        const DML_OPERATOR_DESC splitInputDmlDesc = {DML_OPERATOR_SPLIT, &splitInputDesc};

        // Gather the cos/sin values based on the position ids
        const std::array<uint32_t, 4> gatheredCosSinShape {1, 1, sequenceLength, headSize / 2};
        TensorDesc gatheredCosSinTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, gatheredCosSinShape);
        const DML_TENSOR_DESC gatheredCosSinDmlTensorDesc = gatheredCosSinTensorDesc.GetDmlDesc();

        DML_GATHER_OPERATOR_DESC gatherCosSinDesc{};
        gatherCosSinDesc.InputTensor = &inputDescs[cosCacheIndex];
        gatherCosSinDesc.IndicesTensor = &inputDescs[positionIdsIndex];
        gatherCosSinDesc.OutputTensor = &gatheredCosSinDmlTensorDesc;
        gatherCosSinDesc.Axis = 2;
        gatherCosSinDesc.IndexDimensions = 2;
        const DML_OPERATOR_DESC gatherCosSinDmlDesc {DML_OPERATOR_GATHER, &gatherCosSinDesc};

        // After gathering cos/sin, reshape and broadcast them to match the number of heads of the half input data
        const std::array<uint32_t, 4> reshapedCosSinShape {1, sequenceLength, 1, headSize / 2};
        const std::array<uint32_t, 4> broadcastedCosSinShape {batchSize, sequenceLength, numHeads, headSize / 2};
        TensorDesc broadcastedCosSinTensorDesc = TensorDesc::ConstructBroadcastedTensorDesc(dataType, broadcastedCosSinShape, reshapedCosSinShape);
        const DML_TENSOR_DESC broadcastedCosSinDmlTensorDesc = broadcastedCosSinTensorDesc.GetDmlDesc();

        // Multiply the first half of the data with the cos and the second half of the negated data with the sin
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mulHalfDataDesc{};
        mulHalfDataDesc.ATensor = &splitInputDataDmlTensorDescs.front();
        mulHalfDataDesc.BTensor = &broadcastedCosSinDmlTensorDesc;
        mulHalfDataDesc.OutputTensor = &splitInputDataDmlTensorDescs.front();
        const DML_OPERATOR_DESC mulHalfDataDmlDesc {DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mulHalfDataDesc};

        // Negate the second half of the data
        DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC negateHalfDataDesc{};
        negateHalfDataDesc.InputTensor = &splitInputDataDmlTensorDescs.front();
        negateHalfDataDesc.OutputTensor = &splitInputDataDmlTensorDescs.front();
        const DML_OPERATOR_DESC negateHalfDataDmlDesc {DML_OPERATOR_ELEMENT_WISE_NEGATE, &negateHalfDataDesc};

        // Add the multiplied 2 halves together
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC addHalfDataDesc{};
        addHalfDataDesc.ATensor = &splitInputDataDmlTensorDescs.front();
        addHalfDataDesc.BTensor = &splitInputDataDmlTensorDescs.front();
        addHalfDataDesc.OutputTensor = &splitInputDataDmlTensorDescs.front();
        const DML_OPERATOR_DESC addHalfDataDmlDesc {DML_OPERATOR_ELEMENT_WISE_ADD, &addHalfDataDesc};

        // Join the 2 halves together
        DML_JOIN_OPERATOR_DESC joinHalfDataDesc{};
        joinHalfDataDesc.InputTensors = splitInputDataDmlTensorDescs.data();
        joinHalfDataDesc.OutputTensor = &inputDataDmlTensorDesc;
        joinHalfDataDesc.Axis = gsl::narrow_cast<uint32_t>(splitInputDataTensorShape.size()) - 1;
        joinHalfDataDesc.InputCount = gsl::narrow_cast<uint32_t>(splitInputDataDmlTensorDescs.size());
        const DML_OPERATOR_DESC joinHalfDataDmlDesc {DML_OPERATOR_JOIN, &joinHalfDataDesc};

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        std::array<const DML_OPERATOR_DESC*, 11> opDescs = {
            &splitInputDmlDesc, // Split the input data
            &gatherCosSinDmlDesc, // Gather cos
            &gatherCosSinDmlDesc, // Gather sin

            &mulHalfDataDmlDesc, // Multiply cos with the first half of the input
            &negateHalfDataDmlDesc, // Negate the second half of the input
            &mulHalfDataDmlDesc, // Multiply sin with the negated second of half of the input
            &addHalfDataDmlDesc, // Add the 2 halves together

            &mulHalfDataDmlDesc, // Multiply cos with the second half of the input
            &mulHalfDataDmlDesc, // Multiply sin with the first half of the input
            &addHalfDataDmlDesc, // Add the 2 halves together

            &joinHalfDataDmlDesc, // Join the halves together
        };

        enum NodeIndex : uint32_t
        {
            splitInputOpIndex,
            gatherCosOpIndex,
            gatherSinOpIndex,

            mulCosFirstHalfOpIndex,
            negateSecondHalfOpIndex,
            mulSinNegatedSecondHalfOpIndex,
            addFirstHalfOpIndex,

            mulCosSecondHalfOpIndex,
            mulSinFirstHalfOpIndex,
            addSecondHalfOpIndex,

            joinOpIndex,
        };

        DML_INPUT_GRAPH_EDGE_DESC inputToSplitEdge = {};
        inputToSplitEdge.GraphInputIndex = inputDataIndex;
        inputToSplitEdge.ToNodeIndex = splitInputOpIndex;
        inputToSplitEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToSplitEdge);

        DML_INPUT_GRAPH_EDGE_DESC cosToGatherEdge = {};
        cosToGatherEdge.GraphInputIndex = cosCacheIndex;
        cosToGatherEdge.ToNodeIndex = gatherCosOpIndex;
        cosToGatherEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(cosToGatherEdge);

        DML_INPUT_GRAPH_EDGE_DESC positionIdsToGatherCosEdge = {};
        positionIdsToGatherCosEdge.GraphInputIndex = positionIdsIndex;
        positionIdsToGatherCosEdge.ToNodeIndex = gatherCosOpIndex;
        positionIdsToGatherCosEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(positionIdsToGatherCosEdge);

        DML_INPUT_GRAPH_EDGE_DESC sinToGatherEdge = {};
        sinToGatherEdge.GraphInputIndex = sinCacheIndex;
        sinToGatherEdge.ToNodeIndex = gatherSinOpIndex;
        sinToGatherEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(sinToGatherEdge);

        DML_INPUT_GRAPH_EDGE_DESC positionIdsToGatherSinEdge = {};
        positionIdsToGatherSinEdge.GraphInputIndex = positionIdsIndex;
        positionIdsToGatherSinEdge.ToNodeIndex = gatherSinOpIndex;
        positionIdsToGatherSinEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(positionIdsToGatherSinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC firstHalfDataToMulCosEdge = {};
        firstHalfDataToMulCosEdge.FromNodeIndex = splitInputOpIndex;
        firstHalfDataToMulCosEdge.FromNodeOutputIndex = 0;
        firstHalfDataToMulCosEdge.ToNodeIndex = mulCosFirstHalfOpIndex;
        firstHalfDataToMulCosEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(firstHalfDataToMulCosEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredCosToMulFirstHalfDataEdge = {};
        gatheredCosToMulFirstHalfDataEdge.FromNodeIndex = gatherCosOpIndex;
        gatheredCosToMulFirstHalfDataEdge.FromNodeOutputIndex = 0;
        gatheredCosToMulFirstHalfDataEdge.ToNodeIndex = mulCosFirstHalfOpIndex;
        gatheredCosToMulFirstHalfDataEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(gatheredCosToMulFirstHalfDataEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC secondHalfDataToNegateEdge = {};
        secondHalfDataToNegateEdge.FromNodeIndex = splitInputOpIndex;
        secondHalfDataToNegateEdge.FromNodeOutputIndex = 1;
        secondHalfDataToNegateEdge.ToNodeIndex = negateSecondHalfOpIndex;
        secondHalfDataToNegateEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(secondHalfDataToNegateEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC negatedSecondHalfDataToMulSinEdge = {};
        negatedSecondHalfDataToMulSinEdge.FromNodeIndex = negateSecondHalfOpIndex;
        negatedSecondHalfDataToMulSinEdge.FromNodeOutputIndex = 0;
        negatedSecondHalfDataToMulSinEdge.ToNodeIndex = mulSinNegatedSecondHalfOpIndex;
        negatedSecondHalfDataToMulSinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(negatedSecondHalfDataToMulSinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredSinToMulNegatedSecondHalfDataEdge = {};
        gatheredSinToMulNegatedSecondHalfDataEdge.FromNodeIndex = gatherSinOpIndex;
        gatheredSinToMulNegatedSecondHalfDataEdge.FromNodeOutputIndex = 0;
        gatheredSinToMulNegatedSecondHalfDataEdge.ToNodeIndex = mulSinNegatedSecondHalfOpIndex;
        gatheredSinToMulNegatedSecondHalfDataEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(gatheredSinToMulNegatedSecondHalfDataEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC firstHalfCosMulToAddEdge = {};
        firstHalfCosMulToAddEdge.FromNodeIndex = mulCosFirstHalfOpIndex;
        firstHalfCosMulToAddEdge.FromNodeOutputIndex = 0;
        firstHalfCosMulToAddEdge.ToNodeIndex = addFirstHalfOpIndex;
        firstHalfCosMulToAddEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(firstHalfCosMulToAddEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC secondHalfSinMulToAddEdge = {};
        secondHalfSinMulToAddEdge.FromNodeIndex = mulSinNegatedSecondHalfOpIndex;
        secondHalfSinMulToAddEdge.FromNodeOutputIndex = 0;
        secondHalfSinMulToAddEdge.ToNodeIndex = addFirstHalfOpIndex;
        secondHalfSinMulToAddEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(secondHalfSinMulToAddEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC secondHalfDataToMulCosEdge = {};
        secondHalfDataToMulCosEdge.FromNodeIndex = splitInputOpIndex;
        secondHalfDataToMulCosEdge.FromNodeOutputIndex = 1;
        secondHalfDataToMulCosEdge.ToNodeIndex = mulCosSecondHalfOpIndex;
        secondHalfDataToMulCosEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(secondHalfDataToMulCosEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredCosToMulSecondHalfDataEdge = {};
        gatheredCosToMulSecondHalfDataEdge.FromNodeIndex = gatherCosOpIndex;
        gatheredCosToMulSecondHalfDataEdge.FromNodeOutputIndex = 0;
        gatheredCosToMulSecondHalfDataEdge.ToNodeIndex = mulCosSecondHalfOpIndex;
        gatheredCosToMulSecondHalfDataEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(gatheredCosToMulSecondHalfDataEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC firstHalfDataToMulSinEdge = {};
        firstHalfDataToMulSinEdge.FromNodeIndex = splitInputOpIndex;
        firstHalfDataToMulSinEdge.FromNodeOutputIndex = 0;
        firstHalfDataToMulSinEdge.ToNodeIndex = mulSinFirstHalfOpIndex;
        firstHalfDataToMulSinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(firstHalfDataToMulSinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredSinToMulFirstHalfDataEdge = {};
        gatheredSinToMulFirstHalfDataEdge.FromNodeIndex = gatherSinOpIndex;
        gatheredSinToMulFirstHalfDataEdge.FromNodeOutputIndex = 0;
        gatheredSinToMulFirstHalfDataEdge.ToNodeIndex = mulSinFirstHalfOpIndex;
        gatheredSinToMulFirstHalfDataEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(gatheredSinToMulFirstHalfDataEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC secondHalfCosMulToAddEdge = {};
        secondHalfCosMulToAddEdge.FromNodeIndex = mulCosSecondHalfOpIndex;
        secondHalfCosMulToAddEdge.FromNodeOutputIndex = 0;
        secondHalfCosMulToAddEdge.ToNodeIndex = addSecondHalfOpIndex;
        secondHalfCosMulToAddEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(secondHalfCosMulToAddEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC firstHalfSinMulToAddEdge = {};
        firstHalfSinMulToAddEdge.FromNodeIndex = mulSinFirstHalfOpIndex;
        firstHalfSinMulToAddEdge.FromNodeOutputIndex = 0;
        firstHalfSinMulToAddEdge.ToNodeIndex = addSecondHalfOpIndex;
        firstHalfSinMulToAddEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(firstHalfSinMulToAddEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC firstAddToJoinEdge = {};
        firstAddToJoinEdge.FromNodeIndex = addFirstHalfOpIndex;
        firstAddToJoinEdge.FromNodeOutputIndex = 0;
        firstAddToJoinEdge.ToNodeIndex = joinOpIndex;
        firstAddToJoinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(firstAddToJoinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC secondAddToJoinEdge = {};
        secondAddToJoinEdge.FromNodeIndex = addSecondHalfOpIndex;
        secondAddToJoinEdge.FromNodeOutputIndex = 0;
        secondAddToJoinEdge.ToNodeIndex = joinOpIndex;
        secondAddToJoinEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(secondAddToJoinEdge);

        DML_OUTPUT_GRAPH_EDGE_DESC joinToOutputEdge = {};
        joinToOutputEdge.FromNodeIndex = joinOpIndex;
        joinToOutputEdge.FromNodeOutputIndex = 0;
        joinToOutputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(joinToOutputEdge);

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();
        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();
        operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodesAsOpDesc = opDescs.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(RotaryEmbedding, DmlOperatorRotaryEmbedding);

} // namespace Dml
