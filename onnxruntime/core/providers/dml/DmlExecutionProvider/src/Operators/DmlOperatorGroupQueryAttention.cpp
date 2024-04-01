// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
class DmlOperatorGroupQueryAttention : public DmlOperator, public GroupQueryAttentionHelper
{
public:
    DmlOperatorGroupQueryAttention(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        GroupQueryAttentionHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        enum InputIndex : uint32_t
        {
            queryIndex,
            keyIndex,
            valueIndex,
            pastKeyIndex,
            pastValueIndex,
            seqLensIndex,
            inputCount,
        };

        enum DmlInputIndex : uint32_t
        {
            dmlQueryIndex,
            dmlKeyIndex,
            dmlValueIndex,
            dmlStackedQueryKeyIndex,
            dmlStackedKeyValueIndex,
            dmlStackedQueryKeyValueIndex,
            dmlBiasIndex,
            dmlMaskIndex,
            dmlRelativePositionBiasIndex,
            dmlPastKeyIndex,
            dmlPastValueIndex,
            dmlPastSequenceLengthsIndex,
            dmlInputCount,
        };

        enum OutputIndex : uint32_t
        {
            outputIndex,
            outputPresentKeyIndex,
            outputPresentValueIndex,
            outputCount,
        };

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() >= 1);

        std::vector<std::optional<uint32_t>> inputIndices(dmlInputCount);
        inputIndices[dmlQueryIndex] = queryIndex;
        inputIndices[dmlKeyIndex] = keyIndex;
        inputIndices[dmlValueIndex] = valueIndex;

        if (kernelCreationContext.GetInputTensorShape(queryIndex)[1] == 1)
        {
            inputIndices[dmlPastSequenceLengthsIndex] = seqLensIndex;
        }

        std::vector<std::optional<uint32_t>> outputIndices = {
            outputIndex,
            outputPresentKeyIndex,
            outputPresentValueIndex,
        };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices, std::nullopt, std::nullopt, 1);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlQueryIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetDimensionCount() == 3);

        const uint32_t queryNumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        const uint32_t kvNumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::KvNumHeads));

        auto querySizes = m_inputTensorDescs[dmlQueryIndex].GetSizes();
        auto keySizes = m_inputTensorDescs[dmlKeyIndex].GetSizes();
        auto valueSizes = m_inputTensorDescs[dmlValueIndex].GetSizes();

        const uint32_t batchSize = querySizes[0];
        const uint32_t sequenceLength = querySizes[1];
        const uint32_t queryHiddenSize = querySizes[2];

        const uint32_t kvSequenceLength = keySizes[1];
        const uint32_t kvHiddenSize = keySizes[2];

        const uint32_t queryHeadSize = queryHiddenSize / queryNumHeads;
        const uint32_t kvHeadSize = kvHiddenSize / kvNumHeads;
        const uint32_t totalSequenceLength = GetTotalSequenceLength();

        // Validate Query dimensions
        ML_CHECK_VALID_ARGUMENT(querySizes[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(querySizes[1] == sequenceLength);
        ML_CHECK_VALID_ARGUMENT(querySizes[2] == queryHiddenSize);

        // Validate Key dimensions
        ML_CHECK_VALID_ARGUMENT(keySizes[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(keySizes[1] == kvSequenceLength);
        ML_CHECK_VALID_ARGUMENT(keySizes[2] == kvHiddenSize);

        // Validate Value dimensions
        ML_CHECK_VALID_ARGUMENT(valueSizes[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(valueSizes[1] == kvSequenceLength);
        ML_CHECK_VALID_ARGUMENT(valueSizes[2] == kvHiddenSize);

        if (sequenceLength == 1)
        {
            // Validate PastSequenceLengths dimensions
            if (m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetDimensionCount() == 1)
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetSizes()[0] == batchSize);
            }
            else
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetDimensionCount() == 2);
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetSizes()[0] == batchSize);
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetSizes()[1] == 1);
            }
        }

        const std::array<uint32_t, 1> pastSequenceLengthsShape = {batchSize};
        auto pastSequenceLengthsDataType = MLOperatorTensorDataType::Int32;
        TensorDesc pastSequenceLengthsTensorDesc = TensorDesc::ConstructDefaultTensorDesc(pastSequenceLengthsDataType, pastSequenceLengthsShape);
        const DML_TENSOR_DESC pastSequenceLengthsDmlTensorDesc = pastSequenceLengthsTensorDesc.GetDmlDesc();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC mhaDesc = {};
        mhaDesc.QueryTensor = &inputDescs[dmlQueryIndex];
        mhaDesc.KeyTensor = &inputDescs[dmlKeyIndex];
        mhaDesc.ValueTensor = &inputDescs[dmlValueIndex];
        mhaDesc.PastSequenceLengthsTensor = &pastSequenceLengthsDmlTensorDesc;
        mhaDesc.OutputTensor = &outputDescs[outputIndex];
        mhaDesc.OutputPresentKeyTensor = &outputDescs[outputPresentKeyIndex];
        mhaDesc.OutputPresentValueTensor = &outputDescs[outputPresentValueIndex];
        mhaDesc.QueryHeadCount = queryNumHeads;
        mhaDesc.KeyValueHeadCount = kvNumHeads;
        mhaDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(queryHeadSize)));
        mhaDesc.MaskFilterValue = -10'000.0f;
        DML_OPERATOR_DESC mhaDmlDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION1, &mhaDesc };

        if (sequenceLength == 1)
        {
            SetDmlOperatorDesc(mhaDmlDesc, kernelCreationContext);
        }
        else
        {
            // The GQA offline fusion does this thing where it sums the number of 1's in the mask to figure out the value of the past sequence.
            // This doesn't work well for the first iteration since, obviously, there are no past sequences and the mask in this case represents
            // only the elements in the initial sequence. To work around this, the CUDA implementation of the operator ignores the value of
            // pastSequenceLengths for the first iteration and acts as if it was 0. This feels like a pretty dirty hack and something that should
            // be polished in the future, but for compatibility with the GQA fusion and the CUDA implementation we do the same thing here. We DO NOT
            // want to do this within DirectML since DirectML should be agnostic w.r.t which iteration it's currently executing MHA for, and such a
            // hack that is likely to be modified in the future shouldn't be enshrined within DirectML. Doing it here is OK because the nature of contrib
            // ops is that they can change at any time.
            DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zeroScalarDesc = {};
            zeroScalarDesc.OutputTensor = &pastSequenceLengthsDmlTensorDesc;
            zeroScalarDesc.ValueDataType = pastSequenceLengthsTensorDesc.GetDmlDataType();
            DML_OPERATOR_DESC zeroScalarDmlDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &zeroScalarDesc };

            std::vector<const DML_OPERATOR_DESC*> opDescs = {
                &zeroScalarDmlDesc,
                &mhaDmlDesc,
            };

            // Construct the graph
            std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
            std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

            // Link the query/key/value inputs to MHA
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
                inputToMhaEdge.GraphInputIndex = i;
                inputToMhaEdge.ToNodeIndex = 1;
                inputToMhaEdge.ToNodeInputIndex = i;
                inputEdges.push_back(inputToMhaEdge);
            }

            // Link the zero scalar to MHA
            DML_INTERMEDIATE_GRAPH_EDGE_DESC zeroScalarToMhaEdge = {};
            zeroScalarToMhaEdge.FromNodeIndex = 0;
            zeroScalarToMhaEdge.FromNodeOutputIndex = 0;
            zeroScalarToMhaEdge.ToNodeIndex = 1;
            zeroScalarToMhaEdge.ToNodeInputIndex = dmlPastSequenceLengthsIndex;
            intermediateEdges.push_back(zeroScalarToMhaEdge);

            // Link MHA's outputs to the graph's outputs
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_OUTPUT_GRAPH_EDGE_DESC mhaToOutputEdge = {};
                mhaToOutputEdge.FromNodeIndex = 1;
                mhaToOutputEdge.FromNodeOutputIndex = i;
                mhaToOutputEdge.GraphOutputIndex = i;
                outputEdges.push_back(mhaToOutputEdge);
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
            SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(GroupQueryAttention, DmlOperatorGroupQueryAttention);
} // namespace Dml
