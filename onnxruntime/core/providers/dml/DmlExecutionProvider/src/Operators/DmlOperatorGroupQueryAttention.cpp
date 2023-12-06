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
        inputIndices[dmlPastKeyIndex] = pastKeyIndex;
        inputIndices[dmlPastValueIndex] = pastValueIndex;
        inputIndices[dmlPastSequenceLengthsIndex] = seqLensIndex;

        std::vector<std::optional<uint32_t>> outputIndices = {
            outputIndex,
            outputPresentKeyIndex,
            outputPresentValueIndex,
        };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices, std::nullopt, std::nullopt, 1);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlQueryIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetDimensionCount() == 1);

        const uint32_t queryNumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        const uint32_t kvNumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::KvNumHeads));
        const uint32_t batchSize = m_inputTensorDescs[dmlQueryIndex].GetSizes()[0];
        const uint32_t sequenceLength = m_inputTensorDescs[dmlQueryIndex].GetSizes()[1];
        const uint32_t queryHiddenSize = m_inputTensorDescs[dmlQueryIndex].GetSizes()[2];
        const uint32_t kvSequenceLength = m_inputTensorDescs[dmlKeyIndex].GetSizes()[1];
        const uint32_t kvHiddenSize = m_inputTensorDescs[dmlKeyIndex].GetSizes()[2];
        const uint32_t pastSequenceLength = m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[2];
        const uint32_t queryHeadSize = queryHiddenSize / queryNumHeads;
        const uint32_t kvHeadSize = kvHiddenSize / kvNumHeads;
        const uint32_t totalSequenceLength = GetTotalSequenceLength();

        // Validate Query dimensions
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlQueryIndex].GetSizes()[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlQueryIndex].GetSizes()[1] == sequenceLength);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlQueryIndex].GetSizes()[2] == queryHiddenSize);

        // Validate Key dimensions
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetSizes()[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetSizes()[1] == kvSequenceLength);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetSizes()[2] == kvHiddenSize);

        // Validate Value dimensions
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetSizes()[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetSizes()[1] == kvSequenceLength);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetSizes()[2] == kvHiddenSize);

        // Validate PastKey dimensions
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[1] == kvNumHeads);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[2] == pastSequenceLength);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[3] == kvHeadSize);

        // Validate PastValue dimensions
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetSizes()[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetSizes()[1] == kvNumHeads);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetSizes()[2] == pastSequenceLength);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetSizes()[3] == kvHeadSize);

        // Validate PastSequenceLengths dimensions
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastSequenceLengthsIndex].GetSizes()[0] == batchSize);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC mhaDesc = {};
        mhaDesc.QueryTensor = &inputDescs[dmlQueryIndex];
        mhaDesc.KeyTensor = &inputDescs[dmlKeyIndex];
        mhaDesc.ValueTensor = &inputDescs[dmlValueIndex];
        mhaDesc.PastKeyTensor = &inputDescs[dmlPastKeyIndex];
        mhaDesc.PastValueTensor = &inputDescs[dmlPastValueIndex];
        mhaDesc.PastSequenceLengthsTensor = &inputDescs[dmlPastSequenceLengthsIndex];
        mhaDesc.OutputTensor = &outputDescs[outputIndex];
        mhaDesc.OutputPresentKeyTensor = &outputDescs[outputPresentKeyIndex];
        mhaDesc.OutputPresentValueTensor = &outputDescs[outputPresentValueIndex];
        mhaDesc.QueryHeadCount = queryNumHeads;
        mhaDesc.KeyValueHeadCount = kvNumHeads;
        mhaDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(queryHeadSize)));

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION1, &mhaDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(GroupQueryAttention, DmlOperatorGroupQueryAttention);
} // namespace Dml
