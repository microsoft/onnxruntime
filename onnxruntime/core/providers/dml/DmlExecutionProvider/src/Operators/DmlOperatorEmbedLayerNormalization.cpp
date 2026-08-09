// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

/*
WordEmbeddings    InputIds    SegmentEmbeddings    SegmentIds    PositionEmbeddings    PositionIds    Gamma    Beta    Mask
     │               │              │                  │                │                   │           │        │      │
     │               │              │                  │                │                   │           │        │      │
     │               │              │                  │                │                   │           │        │      │
     │               │              │                  │                │                   │           │        │      │
     │               │              │                  │                │                   │           │        │      │          OnesConstant
     │               │              │                  │                │                   │           │        │      │                │
     └───────┬───────┘              └────────┬─────────┘                └─────────┬─────────┘           │        │      │                │
             │                               │                                    │                     │        │      │                │
             ▼                               ▼                                    ▼                     │        │      └───────┬────────┘
          Gather                           Gather                              Gather                   │        │              │
             │                               │                                    │                     │        │              ▼
             │                               │                                    │                     │        │            Equals
             │                               │                                    │                     │        │              │
             │                               │                                    │                     │        │              │
             │                               │                                    │                     │        │              │
             │                               │                                    │                     │        │              │
             └─────────────┬─────────────────┘                                    │                     │        │              │
                           │                                                      │                     │        │              │
                           ▼                                                      │                     │        │              │
                          Add                                                     │                     │        │              │
                           │                                                      │                     │        │              ▼
                           │                                                      │                     │        │            Reduce
                           └────────────────────────────┬─────────────────────────┘                     │        │              │
                                                        │                                               │        │              │
                                                        ▼                                               │        │              │
                                                       Add   ┌──────────────────────────────────────────┘        │              │
                                                        │    │                                                   │              │
                           ┌────────────────────────────┤    │    ┌──────────────────────────────────────────────┘              │
                           │                            │    │    │                                                             │
                           │                            ▼    ▼    ▼                                                             │
                           │                MeanVarianceNormalization                                                           │
                           │                            │               ┌───────────────────────────────────────────────────────┘
                           │                            │               │
                           │                            │               │
                           ▼                            ▼               ▼
                      EmbeddingSum                   Output          MaskIndex

 This kernel creates a DML_GRAPH, as mentioned above.
 For reference, refer to this Doc:
 https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
 */

namespace Dml
{

class DmlOperatorEmbedLayerNormalization : public DmlOperator
{
public:
    DmlOperatorEmbedLayerNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<std::optional<uint32_t>> kernelOutputIndices = {0, 1, 2};

        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices, kernelOutputIndices);
        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);

        assert(m_inputTensorDescs.size() == 9);
        assert(m_outputTensorDescs.size() == 3);

        auto inputIdsDesc = m_inputTensorDescs[0].GetDmlDesc();
        auto segmentIdsDesc = m_inputTensorDescs[1].GetDmlDesc();
        auto wordEmbeddingDesc = m_inputTensorDescs[2].GetDmlDesc();
        auto positionEmbeddingDesc = m_inputTensorDescs[3].GetDmlDesc();
        auto segmentEmbeddingDesc = m_inputTensorDescs[4].GetDmlDesc();
        auto gammaDesc = m_inputTensorDescs[5].GetDmlDesc();
        auto betaDesc = m_inputTensorDescs[6].GetDmlDesc();
        auto maskDesc = m_inputTensorDescs[7].GetDmlDesc();
        auto positionIdsDesc = m_inputTensorDescs[8].GetDmlDesc();
        auto outputDesc = m_outputTensorDescs[0].GetDmlDesc();
        auto maskIndexDesc = m_outputTensorDescs[1].GetDmlDesc();
        auto embeddingSumDesc = m_outputTensorDescs[2].GetDmlDesc();

        const DML_TENSOR_DATA_TYPE indicesDataType = m_inputTensorDescs[0].GetDmlDataType();
        const DML_TENSOR_DATA_TYPE valuesDataType = m_inputTensorDescs[2].GetDmlDataType();
        const uint32_t batchSize = m_inputTensorDescs[0].GetSizes()[2];
        const uint32_t sequenceLength = m_inputTensorDescs[0].GetSizes()[3];

        // When position ids are not given, the indices are simply the sequence ids in ascending order
        TensorDesc positionSequenceIdsTensorDesc(indicesDataType, std::vector<uint32_t>({1u, 1u, 1u, sequenceLength}));
        DML_TENSOR_DESC positionSequenceIdsDmlTensorDesc = positionSequenceIdsTensorDesc.GetDmlDesc();

        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC positionSequenceIdsDesc = {};
        positionSequenceIdsDesc.ValueStart.Int32 = 0;
        positionSequenceIdsDesc.ValueDelta.Int32 = 1;
        positionSequenceIdsDesc.ValueDataType = indicesDataType;
        positionSequenceIdsDesc.OutputTensor = &positionSequenceIdsDmlTensorDesc;
        DML_OPERATOR_DESC positionSequenceIdsOpDesc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &positionSequenceIdsDesc };

        // Gather the word embeddings
        TensorDesc gatheredTensorDesc(valuesDataType, m_outputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC gatheredDmlTensorDesc = gatheredTensorDesc.GetDmlDesc();

        DML_GATHER_OPERATOR_DESC wordEmbeddingGatherDesc = {};
        wordEmbeddingGatherDesc.InputTensor = &wordEmbeddingDesc;
        wordEmbeddingGatherDesc.IndicesTensor = &inputIdsDesc;
        wordEmbeddingGatherDesc.OutputTensor = &gatheredDmlTensorDesc;
        wordEmbeddingGatherDesc.Axis = 2;
        wordEmbeddingGatherDesc.IndexDimensions = 2;
        DML_OPERATOR_DESC wordEmbeddingGatherOpDesc = { DML_OPERATOR_GATHER, &wordEmbeddingGatherDesc };

        // Gather the position embeddings
        std::optional<std::vector<uint32_t>> positionIdsStrides;
        if (positionIdsDesc.Desc && m_inputTensorDescs[8].GetSizes()[2] == 1 || !positionIdsDesc.Desc)
        {
            positionIdsStrides = std::vector<uint32_t>({0, 0, 0, 1});
        }

        TensorDesc positionIdsTensorDesc(indicesDataType, m_inputTensorDescs[0].GetSizes(), std::move(positionIdsStrides));
        DML_TENSOR_DESC positionIdsDmlTensorDesc = positionIdsTensorDesc.GetDmlDesc();

        DML_GATHER_OPERATOR_DESC positionEmbeddingGatherDesc = {};
        positionEmbeddingGatherDesc.InputTensor = &positionEmbeddingDesc;
        positionEmbeddingGatherDesc.IndicesTensor = &positionIdsDmlTensorDesc;
        positionEmbeddingGatherDesc.OutputTensor = &gatheredDmlTensorDesc;
        positionEmbeddingGatherDesc.Axis = 2;
        positionEmbeddingGatherDesc.IndexDimensions = 2;
        DML_OPERATOR_DESC positionEmbeddingGatherOpDesc = { DML_OPERATOR_GATHER, &positionEmbeddingGatherDesc };

        // Gather the segment embeddings
        DML_GATHER_OPERATOR_DESC segmentEmbeddingGatherDesc = {};
        segmentEmbeddingGatherDesc.InputTensor = &segmentEmbeddingDesc;
        segmentEmbeddingGatherDesc.IndicesTensor = &segmentIdsDesc;
        segmentEmbeddingGatherDesc.OutputTensor = &gatheredDmlTensorDesc;
        segmentEmbeddingGatherDesc.Axis = 2;
        segmentEmbeddingGatherDesc.IndexDimensions = 2;
        DML_OPERATOR_DESC segmentEmbeddingGatherOpDesc = { DML_OPERATOR_GATHER, &segmentEmbeddingGatherDesc };

        // Add the embeddings together
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC embeddingsAddDesc = {};
        embeddingsAddDesc.ATensor = &gatheredDmlTensorDesc;
        embeddingsAddDesc.BTensor = &gatheredDmlTensorDesc;
        embeddingsAddDesc.OutputTensor = &gatheredDmlTensorDesc;
        DML_OPERATOR_DESC embeddingsAddOpDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &embeddingsAddDesc };

        // Execute MVN
        std::vector<uint32_t> mvnReductionAxes({m_inputTensorDescs[0].GetDimensionCount() - 1});

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvnDesc = {};
        mvnDesc.InputTensor = &gatheredDmlTensorDesc;
        mvnDesc.ScaleTensor = &gammaDesc;
        mvnDesc.BiasTensor = &betaDesc;
        mvnDesc.OutputTensor = &outputDesc;
        mvnDesc.Axes = mvnReductionAxes.data();
        mvnDesc.AxisCount = gsl::narrow_cast<uint32_t>(mvnReductionAxes.size());
        mvnDesc.NormalizeVariance = true;
        mvnDesc.Epsilon = epsilon;
        mvnDesc.FusedActivation = nullptr;
        DML_OPERATOR_DESC mvnOpDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &mvnDesc };

        TensorDesc scalarTensorDesc(indicesDataType, std::vector<uint32_t>(m_inputTensorDescs[0].GetDimensionCount(), 1));
        DML_TENSOR_DESC scalarDmlTensorDesc = scalarTensorDesc.GetDmlDesc();

        // Create a tensor full of zeros
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zerosDesc = {};
        zerosDesc.Value.Int32 = 0;
        zerosDesc.ValueDataType = indicesDataType;
        zerosDesc.OutputTensor = &scalarDmlTensorDesc;
        DML_OPERATOR_DESC zerosOpDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &zerosDesc };

        // Create a tensor full of ones
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC onesDesc = {};
        onesDesc.Value.Int32 = 1;
        onesDesc.ValueDataType = indicesDataType;
        onesDesc.OutputTensor = &scalarDmlTensorDesc;
        DML_OPERATOR_DESC onesOpDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &onesDesc };

        TensorDesc broadcastedOnesTensorDesc(indicesDataType, m_inputTensorDescs[0].GetSizes(), std::vector<uint32_t>(m_inputTensorDescs[0].GetDimensionCount()));
        DML_TENSOR_DESC broadcastedOnesDmlTensorDesc = broadcastedOnesTensorDesc.GetDmlDesc();

        // Create the equal operator to keep all values in the mask that are 1
        TensorDesc equalOutputTensorDesc(DML_TENSOR_DATA_TYPE_UINT32, m_inputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC equalOutputDmlTensorDesc = equalOutputTensorDesc.GetDmlDesc();

        DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC equalDesc = {};
        equalDesc.ATensor = &maskDesc;
        equalDesc.BTensor = &broadcastedOnesDmlTensorDesc;
        equalDesc.OutputTensor = &equalOutputDmlTensorDesc;
        DML_OPERATOR_DESC equalOpDesc = { DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS, &equalDesc };

        // Reinterpret the uint32 tensor to an int32 tensor
        TensorDesc sparseMaskTensorDesc(indicesDataType, m_inputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC sparseMaskDmlTensorDesc = sparseMaskTensorDesc.GetDmlDesc();

        // Create the reduce operator to sum the values of the mask for each batch
        TensorDesc reducedMaskTensorDesc(indicesDataType, std::vector<uint32_t>({1, 1, batchSize, 1}));
        DML_TENSOR_DESC reducedMaskDmlTensorDesc = reducedMaskTensorDesc.GetDmlDesc();

        uint32_t reduceAxes[] = {3};
        DML_REDUCE_OPERATOR_DESC reduceDesc = {};
        reduceDesc.Axes = reduceAxes;
        reduceDesc.AxisCount = 1;
        reduceDesc.Function = DML_REDUCE_FUNCTION_SUM;
        reduceDesc.InputTensor = &sparseMaskDmlTensorDesc;
        reduceDesc.OutputTensor = &reducedMaskDmlTensorDesc;
        DML_OPERATOR_DESC reduceOpDesc = { DML_OPERATOR_REDUCE, &reduceDesc };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        opDescs.reserve(11);

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(9);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(8);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(3);

        uint32_t currentNodeIndex = 0;

        // Insert the zeros operation into the graph
        const uint32_t zerosNodeIndex = currentNodeIndex;
        if (!maskDesc.Desc)
        {
            opDescs.push_back(&zerosOpDesc);
            currentNodeIndex++;
        }

        // Insert the sequence operation into the graph
        const uint32_t sequenceIdsNodeIndex = currentNodeIndex;
        if (!positionIdsDesc.Desc)
        {
            opDescs.push_back(&positionSequenceIdsOpDesc);
            currentNodeIndex++;
        }

        // Insert the word embeddings Gather operation into the graph
        const uint32_t gatherWordsNodeIndex = currentNodeIndex;
        opDescs.push_back(&wordEmbeddingGatherOpDesc);
        currentNodeIndex++;

        // Insert the position embeddings Gather operation into the graph
        const uint32_t gatherPositionsNodeIndex = currentNodeIndex;
        opDescs.push_back(&positionEmbeddingGatherOpDesc);
        currentNodeIndex++;

        // Insert the segment embeddings Gather opetation into the graph
        const uint32_t gatherSegmentsNodeIndex = currentNodeIndex;
        if (segmentEmbeddingDesc.Desc)
        {
            opDescs.push_back(&segmentEmbeddingGatherOpDesc);
            currentNodeIndex++;
        }

        // Insert the word+position embeddings operation into the graph
        const uint32_t wordsPositionsAddNodeIndex = currentNodeIndex;
        opDescs.push_back(&embeddingsAddOpDesc);
        currentNodeIndex++;

        // Insert the word+position+segment embeddings operation into the graph
        const uint32_t wordsPositionsSegmentsAddNodeIndex = currentNodeIndex;
        if (segmentEmbeddingDesc.Desc)
        {
            opDescs.push_back(&embeddingsAddOpDesc);
            currentNodeIndex++;
        }

        // Insert the MVN operation into the graph
        const uint32_t mvnNodeIndex = currentNodeIndex;
        opDescs.push_back(&mvnOpDesc);
        currentNodeIndex++;

        // Insert the Ones operation into the graph
        const uint32_t onesNodeIndex = currentNodeIndex;
        if (maskDesc.Desc)
        {
            opDescs.push_back(&onesOpDesc);
            currentNodeIndex++;
        }

        // Insert the Equal operation into the graph
        const uint32_t equalNodeIndex = currentNodeIndex;
        if (maskDesc.Desc)
        {
            opDescs.push_back(&equalOpDesc);
            currentNodeIndex++;
        }

        // Insert the Reduce operation into the graph
        const uint32_t reduceNodeIndex = currentNodeIndex;
        if (maskDesc.Desc)
        {
            opDescs.push_back(&reduceOpDesc);
            currentNodeIndex++;
        }

        // Insert the edges feeding into the words' gather operation
        DML_INPUT_GRAPH_EDGE_DESC wordEmbeddingsInputEdge = {};
        wordEmbeddingsInputEdge.GraphInputIndex = 2;
        wordEmbeddingsInputEdge.ToNodeIndex = gatherWordsNodeIndex;
        wordEmbeddingsInputEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(std::move(wordEmbeddingsInputEdge));

        DML_INPUT_GRAPH_EDGE_DESC inputIdsInputEdge = {};
        inputIdsInputEdge.GraphInputIndex = 0;
        inputIdsInputEdge.ToNodeIndex = gatherWordsNodeIndex;
        inputIdsInputEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(std::move(inputIdsInputEdge));

        // Insert the edges feeding into the positions' gather operation
        DML_INPUT_GRAPH_EDGE_DESC positionEmbeddingsInputEdge = {};
        positionEmbeddingsInputEdge.GraphInputIndex = 3;
        positionEmbeddingsInputEdge.ToNodeIndex = gatherPositionsNodeIndex;
        positionEmbeddingsInputEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(std::move(positionEmbeddingsInputEdge));

        if (positionIdsDesc.Desc)
        {
            DML_INPUT_GRAPH_EDGE_DESC positionIdsInputEdge = {};
            positionIdsInputEdge.GraphInputIndex = 8;
            positionIdsInputEdge.ToNodeIndex = gatherPositionsNodeIndex;
            positionIdsInputEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(std::move(positionIdsInputEdge));
        }
        else
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC sequenceIntermediateEdge = {};
            sequenceIntermediateEdge.FromNodeIndex = sequenceIdsNodeIndex;
            sequenceIntermediateEdge.FromNodeOutputIndex = 0;
            sequenceIntermediateEdge.ToNodeIndex = gatherPositionsNodeIndex;
            sequenceIntermediateEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(std::move(sequenceIntermediateEdge));
        }

        // Insert the edges feeding into the word+position operation
        DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredWordsIntermediateEdge = {};
        gatheredWordsIntermediateEdge.FromNodeIndex = gatherWordsNodeIndex;
        gatheredWordsIntermediateEdge.FromNodeOutputIndex = 0;
        gatheredWordsIntermediateEdge.ToNodeIndex = wordsPositionsAddNodeIndex;
        gatheredWordsIntermediateEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(std::move(gatheredWordsIntermediateEdge));

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredPositionsIntermediateEdge = {};
        gatheredPositionsIntermediateEdge.FromNodeIndex = gatherPositionsNodeIndex;
        gatheredPositionsIntermediateEdge.FromNodeOutputIndex = 0;
        gatheredPositionsIntermediateEdge.ToNodeIndex = wordsPositionsAddNodeIndex;
        gatheredPositionsIntermediateEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(std::move(gatheredPositionsIntermediateEdge));

        if (segmentEmbeddingDesc.Desc)
        {
            // Insert the edges feeding into the segments' gather operation
            DML_INPUT_GRAPH_EDGE_DESC segmentEmbeddingsInputEdge = {};
            segmentEmbeddingsInputEdge.GraphInputIndex = 4;
            segmentEmbeddingsInputEdge.ToNodeIndex = gatherSegmentsNodeIndex;
            segmentEmbeddingsInputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(std::move(segmentEmbeddingsInputEdge));

            DML_INPUT_GRAPH_EDGE_DESC segmentIdsInputEdge = {};
            segmentIdsInputEdge.GraphInputIndex = 1;
            segmentIdsInputEdge.ToNodeIndex = gatherSegmentsNodeIndex;
            segmentIdsInputEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(std::move(segmentIdsInputEdge));

            // Insert the edges feeding into the word+position+segment operation
            DML_INTERMEDIATE_GRAPH_EDGE_DESC wordsPositionsAddIntermediateEdge = {};
            wordsPositionsAddIntermediateEdge.FromNodeIndex = wordsPositionsAddNodeIndex;
            wordsPositionsAddIntermediateEdge.FromNodeOutputIndex = 0;
            wordsPositionsAddIntermediateEdge.ToNodeIndex = wordsPositionsSegmentsAddNodeIndex;
            wordsPositionsAddIntermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(wordsPositionsAddIntermediateEdge));

            DML_INTERMEDIATE_GRAPH_EDGE_DESC gatheredSegmentsIntermediateEdge = {};
            gatheredSegmentsIntermediateEdge.FromNodeIndex = gatherSegmentsNodeIndex;
            gatheredSegmentsIntermediateEdge.FromNodeOutputIndex = 0;
            gatheredSegmentsIntermediateEdge.ToNodeIndex = wordsPositionsSegmentsAddNodeIndex;
            gatheredSegmentsIntermediateEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(std::move(gatheredSegmentsIntermediateEdge));

            // Insert the edges feeding into the MVN operation
            DML_INTERMEDIATE_GRAPH_EDGE_DESC wordsPositionsSegmentsAddIntermediateEdge = {};
            wordsPositionsSegmentsAddIntermediateEdge.FromNodeIndex = wordsPositionsSegmentsAddNodeIndex;
            wordsPositionsSegmentsAddIntermediateEdge.FromNodeOutputIndex = 0;
            wordsPositionsSegmentsAddIntermediateEdge.ToNodeIndex = mvnNodeIndex;
            wordsPositionsSegmentsAddIntermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(wordsPositionsSegmentsAddIntermediateEdge));

            if (embeddingSumDesc.Desc)
            {
                // Insert the edge feeding into the EmbeddingSum output
                DML_OUTPUT_GRAPH_EDGE_DESC embeddingSumOutputEdge = {};
                embeddingSumOutputEdge.GraphOutputIndex = 2;
                embeddingSumOutputEdge.FromNodeIndex = wordsPositionsSegmentsAddNodeIndex;
                embeddingSumOutputEdge.FromNodeOutputIndex = 0;
                outputEdges.push_back(std::move(embeddingSumOutputEdge));
            }
        }
        else
        {
            // Insert the edges feeding into the MVN operation
            DML_INTERMEDIATE_GRAPH_EDGE_DESC wordsPositionsAddIntermediateEdge = {};
            wordsPositionsAddIntermediateEdge.FromNodeIndex = wordsPositionsAddNodeIndex;
            wordsPositionsAddIntermediateEdge.FromNodeOutputIndex = 0;
            wordsPositionsAddIntermediateEdge.ToNodeIndex = mvnNodeIndex;
            wordsPositionsAddIntermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(wordsPositionsAddIntermediateEdge));

            if (embeddingSumDesc.Desc)
            {
                // Insert the edge feeding into the EmbeddingSum output
                DML_OUTPUT_GRAPH_EDGE_DESC embeddingSumOutputEdge = {};
                embeddingSumOutputEdge.GraphOutputIndex = 2;
                embeddingSumOutputEdge.FromNodeIndex = wordsPositionsAddNodeIndex;
                embeddingSumOutputEdge.FromNodeOutputIndex = 0;
                outputEdges.push_back(std::move(embeddingSumOutputEdge));
            }
        }

        // Insert the remaining edges feeding into the MVN operation
        DML_INPUT_GRAPH_EDGE_DESC gammaInputEdge = {};
        gammaInputEdge.GraphInputIndex = 5;
        gammaInputEdge.ToNodeIndex = mvnNodeIndex;
        gammaInputEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(std::move(gammaInputEdge));

        DML_INPUT_GRAPH_EDGE_DESC betaInputEdge = {};
        betaInputEdge.GraphInputIndex = 6;
        betaInputEdge.ToNodeIndex = mvnNodeIndex;
        betaInputEdge.ToNodeInputIndex = 2;
        inputEdges.push_back(std::move(betaInputEdge));

        if (maskDesc.Desc)
        {
            // Insert the edges feeding into the Equal operation
            DML_INPUT_GRAPH_EDGE_DESC maskInputEdge = {};
            maskInputEdge.GraphInputIndex = 7;
            maskInputEdge.ToNodeIndex = equalNodeIndex;
            maskInputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(std::move(maskInputEdge));

            DML_INTERMEDIATE_GRAPH_EDGE_DESC onesIntermediateEdge = {};
            onesIntermediateEdge.FromNodeIndex = onesNodeIndex;
            onesIntermediateEdge.FromNodeOutputIndex = 0;
            onesIntermediateEdge.ToNodeIndex = equalNodeIndex;
            onesIntermediateEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(std::move(onesIntermediateEdge));

            // Insert the edges feeding into the Reduce operation
            DML_INTERMEDIATE_GRAPH_EDGE_DESC equalIntermediateEdge = {};
            equalIntermediateEdge.FromNodeIndex = equalNodeIndex;
            equalIntermediateEdge.FromNodeOutputIndex = 0;
            equalIntermediateEdge.ToNodeIndex = reduceNodeIndex;
            equalIntermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(equalIntermediateEdge));

            // Insert the edge feeding into the MaskIndex output
            DML_OUTPUT_GRAPH_EDGE_DESC maskIndexOutputEdge = {};
            maskIndexOutputEdge.GraphOutputIndex = 1;
            maskIndexOutputEdge.FromNodeIndex = reduceNodeIndex;
            maskIndexOutputEdge.FromNodeOutputIndex = 0;
            outputEdges.push_back(std::move(maskIndexOutputEdge));
        }
        else if (maskIndexDesc.Desc)
        {
            // Insert the edge feeding into the MaskIndex output
            DML_OUTPUT_GRAPH_EDGE_DESC maskIndexOutputEdge = {};
            maskIndexOutputEdge.GraphOutputIndex = 1;
            maskIndexOutputEdge.FromNodeIndex = zerosNodeIndex;
            maskIndexOutputEdge.FromNodeOutputIndex = 0;
            outputEdges.push_back(std::move(maskIndexOutputEdge));
        }

        // Insert the edge feeding into the values output
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.GraphOutputIndex = 0;
        outputEdge.FromNodeIndex = mvnNodeIndex;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdges.push_back(std::move(outputEdge));

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();
        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();
        operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodes = opDescs.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(EmbedLayerNormalization, DmlOperatorEmbedLayerNormalization);

} // namespace Dml
