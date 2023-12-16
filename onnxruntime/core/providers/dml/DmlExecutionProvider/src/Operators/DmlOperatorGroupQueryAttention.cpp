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

        const std::array<uint32_t, 1> pastSequenceLengthsShape = {batchSize};
        auto pastSequenceLengthsDataType = kernelCreationContext.GetInputEdgeDescription(seqLensIndex).tensorDataType;
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

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION1, &mhaDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(GroupQueryAttention, DmlOperatorGroupQueryAttention);
} // namespace Dml
