// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
class DmlOperatorMultiHeadAttention : public DmlOperator
{
public:
    DmlOperatorMultiHeadAttention(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        enum InputIndex : uint32_t
        {
            queryIndex,
            keyIndex,
            valueIndex,
            biasIndex,
            maskIndex,
            relativePositionBiasIndex,
            pastKeyIndex,
            pastValueIndex,
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

        const bool keyValueIsPast = kernelCreationContext.IsInputValid(keyIndex) && kernelCreationContext.GetInputTensorDimensionCount(keyIndex) == 4;
        const bool hasValue = kernelCreationContext.IsInputValid(valueIndex) && !keyValueIsPast;
        const bool hasBias = kernelCreationContext.IsInputValid(biasIndex);
        const bool hasMask = kernelCreationContext.IsInputValid(maskIndex);
        const bool hasRelativePositionBias = kernelCreationContext.IsInputValid(relativePositionBiasIndex);
        const bool hasPastKey = keyValueIsPast || kernelCreationContext.IsInputValid(pastKeyIndex);
        const bool hasPastValue = keyValueIsPast || kernelCreationContext.IsInputValid(pastValueIndex);
        const bool hasPresentKeyOutput = kernelCreationContext.IsOutputValid(outputPresentKeyIndex);
        const bool hasPresentValueOutput = kernelCreationContext.IsOutputValid(outputPresentValueIndex);
        const bool stackedQkv = kernelCreationContext.GetInputTensorDimensionCount(queryIndex) == 5;
        const bool stackedKv = kernelCreationContext.IsInputValid(keyIndex) && kernelCreationContext.GetInputTensorDimensionCount(keyIndex) == 5;
        const bool hasKey = !stackedKv && !keyValueIsPast && kernelCreationContext.IsInputValid(keyIndex);

        std::vector<std::optional<uint32_t>> inputIndices = {
            stackedQkv ? std::nullopt : std::optional<uint32_t>(queryIndex),
            hasKey ? std::optional<uint32_t>(keyIndex) : std::nullopt,
            hasValue ? std::optional<uint32_t>(valueIndex) : std::nullopt,
            std::nullopt,
            stackedKv ? std::optional<uint32_t>(keyIndex) : std::nullopt,
            stackedQkv ? std::optional<uint32_t>(queryIndex) : std::nullopt,
            biasIndex,
            hasMask ? std::optional<uint32_t>(maskIndex) : std::nullopt,
            relativePositionBiasIndex,
            keyValueIsPast ? keyIndex : pastKeyIndex,
            keyValueIsPast ? valueIndex : pastValueIndex,
        };

        std::vector<std::optional<uint32_t>> outputIndices = {
            outputIndex,
            outputPresentKeyIndex,
            outputPresentValueIndex,
        };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices, std::nullopt, std::nullopt, 1);

        ML_CHECK_VALID_ARGUMENT(!stackedQkv || m_inputTensorDescs[dmlStackedQueryKeyValueIndex].GetDimensionCount() == 5);
        ML_CHECK_VALID_ARGUMENT(stackedQkv || m_inputTensorDescs[dmlQueryIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(!hasKey || m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(!hasValue || m_inputTensorDescs[dmlValueIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(!hasPastKey || m_inputTensorDescs[dmlPastKeyIndex].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(!hasPastValue || m_inputTensorDescs[dmlPastValueIndex].GetDimensionCount() == 4);

        const uint32_t batchSize = stackedQkv
            ? m_inputTensorDescs[dmlStackedQueryKeyValueIndex].GetSizes()[0]
            : m_inputTensorDescs[dmlQueryIndex].GetSizes()[0];

        const uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        const uint32_t headSize = stackedQkv
            ? m_inputTensorDescs[dmlStackedQueryKeyValueIndex].GetSizes()[4]
            : m_inputTensorDescs[dmlQueryIndex].GetSizes()[2] / numHeads;

        const uint32_t sequenceLength = stackedQkv
            ? m_inputTensorDescs[dmlStackedQueryKeyValueIndex].GetSizes()[1]
            : m_inputTensorDescs[dmlQueryIndex].GetSizes()[1];

        uint32_t kvSequenceLength;
        if (hasKey)
        {
            kvSequenceLength = m_inputTensorDescs[dmlKeyIndex].GetSizes()[1];
        }
        else if (stackedKv)
        {
            kvSequenceLength = m_inputTensorDescs[dmlStackedKeyValueIndex].GetSizes()[1];
        }
        else if (hasPastKey)
        {
            kvSequenceLength = m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[2];
        }
        else
        {
            kvSequenceLength = sequenceLength;
        }

        const uint32_t hiddenSize = numHeads * headSize;
        const uint32_t vHiddenSize = hasValue ? m_inputTensorDescs[dmlValueIndex].GetSizes()[2] : hiddenSize;
        const uint32_t pastSequenceLength = hasPastKey ? m_inputTensorDescs[dmlPastKeyIndex].GetSizes()[2] : 0;
        const uint32_t totalSequenceLength = kvSequenceLength + pastSequenceLength;

        if (stackedQkv)
        {
            auto stackedQkvSizes = m_inputTensorDescs[dmlStackedQueryKeyValueIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(stackedQkvSizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(stackedQkvSizes[1] == sequenceLength);
            ML_CHECK_VALID_ARGUMENT(stackedQkvSizes[2] == numHeads);
            ML_CHECK_VALID_ARGUMENT(stackedQkvSizes[3] == 3);
            ML_CHECK_VALID_ARGUMENT(stackedQkvSizes[4] == headSize);
        }
        else
        {
            auto querySizes = m_inputTensorDescs[dmlQueryIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(querySizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(querySizes[1] == sequenceLength);
            ML_CHECK_VALID_ARGUMENT(querySizes[2] == hiddenSize);
        }

        if (hasKey)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 3);

            auto keySizes = m_inputTensorDescs[dmlKeyIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(keySizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(keySizes[1] == kvSequenceLength);
            ML_CHECK_VALID_ARGUMENT(keySizes[2] == hiddenSize);
        }

        if (hasValue)
        {
            auto valueSizes = m_inputTensorDescs[dmlValueIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(valueSizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(valueSizes[1] == kvSequenceLength);
            ML_CHECK_VALID_ARGUMENT(valueSizes[2] == vHiddenSize);
        }

        if (stackedKv)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlStackedKeyValueIndex].GetDimensionCount() == 5);

            auto stackedKvSizes = m_inputTensorDescs[dmlStackedKeyValueIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(stackedKvSizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(stackedKvSizes[1] == kvSequenceLength);
            ML_CHECK_VALID_ARGUMENT(stackedKvSizes[2] == numHeads);
            ML_CHECK_VALID_ARGUMENT(stackedKvSizes[3] == 2);
            ML_CHECK_VALID_ARGUMENT(stackedKvSizes[4] == headSize);
        }

        if (hasBias)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlBiasIndex].GetDimensionCount() == 1);
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlBiasIndex].GetSizes()[0] == hiddenSize + hiddenSize + vHiddenSize);
        }

        DML_MULTIHEAD_ATTENTION_MASK_TYPE maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        if (hasMask)
        {
            if (kernelCreationContext.GetInputTensorDimensionCount(maskIndex) == 1)
            {
                const auto unpaddedKeyBoundsShape = m_inputTensorDescs[dmlMaskIndex].GetSizes();
                ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape.size() == 1);
                ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape[0] == batchSize || unpaddedKeyBoundsShape[0] == batchSize * 3 + 2);

                maskType = unpaddedKeyBoundsShape[0] == batchSize
                    ? DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH
                    : DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_QUERY_SEQUENCE_LENGTH_START_END;

                if (maskType == DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH)
                {
                    uint32_t desiredShape[2] = {1, batchSize};
                    m_inputTensorDescs[dmlMaskIndex] = TensorDesc(
                        m_inputTensorDescs[dmlMaskIndex].GetDmlDataType(),
                        desiredShape);
                }
            }
            else
            {
                const auto keyPaddingMaskTensorShape = m_inputTensorDescs[dmlMaskIndex].GetSizes();
                ML_CHECK_VALID_ARGUMENT(keyPaddingMaskTensorShape.size() == 2);
                ML_CHECK_VALID_ARGUMENT(keyPaddingMaskTensorShape[0] == batchSize);
                ML_CHECK_VALID_ARGUMENT(keyPaddingMaskTensorShape[1] == kvSequenceLength);

                const uint32_t actualShape[4] = {batchSize, 1, 1, kvSequenceLength};
                const uint32_t desiredShape[4] = {batchSize, numHeads, sequenceLength, kvSequenceLength};

                m_inputTensorDescs[dmlMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(
                    m_inputTensorDescs[dmlMaskIndex].GetMlOperatorDataType(),
                    desiredShape,
                    actualShape);

                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
            }
        }

        if (hasRelativePositionBias)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlRelativePositionBiasIndex].GetDimensionCount() == 4);

            auto relativePositionBiasSizes = m_inputTensorDescs[dmlRelativePositionBiasIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasSizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasSizes[1] == numHeads);
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasSizes[2] == sequenceLength);
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasSizes[3] == totalSequenceLength);
        }

        if (hasPastKey)
        {
            auto pastKeySizes = m_inputTensorDescs[dmlPastKeyIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(pastKeySizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(pastKeySizes[1] == numHeads);
            ML_CHECK_VALID_ARGUMENT(pastKeySizes[2] == pastSequenceLength);
            ML_CHECK_VALID_ARGUMENT(pastKeySizes[3] == headSize);
        }

        if (hasPastValue)
        {
            auto pastValueSizes = m_inputTensorDescs[dmlPastValueIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(pastValueSizes[0] == batchSize);
            ML_CHECK_VALID_ARGUMENT(pastValueSizes[1] == numHeads);
            ML_CHECK_VALID_ARGUMENT(pastValueSizes[2] == pastSequenceLength);
            ML_CHECK_VALID_ARGUMENT(pastValueSizes[3] == headSize);
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mhaDesc = {};
        mhaDesc.QueryTensor = stackedQkv ? nullptr : &inputDescs[dmlQueryIndex];
        mhaDesc.KeyTensor = hasKey ? &inputDescs[dmlKeyIndex] : nullptr;
        mhaDesc.ValueTensor = hasValue ? &inputDescs[dmlValueIndex] : nullptr;
        mhaDesc.StackedKeyValueTensor = stackedKv ? &inputDescs[dmlStackedKeyValueIndex] : nullptr;
        mhaDesc.StackedQueryKeyValueTensor = stackedQkv ? &inputDescs[dmlStackedQueryKeyValueIndex] : nullptr;
        mhaDesc.BiasTensor = hasBias ? &inputDescs[dmlBiasIndex] : nullptr;
        mhaDesc.MaskTensor = hasMask ? &inputDescs[dmlMaskIndex] : nullptr;
        mhaDesc.RelativePositionBiasTensor = hasRelativePositionBias ? &inputDescs[dmlRelativePositionBiasIndex] : nullptr;
        mhaDesc.PastKeyTensor = hasPastKey ? &inputDescs[dmlPastKeyIndex] : nullptr;
        mhaDesc.PastValueTensor = hasPastValue ? &inputDescs[dmlPastValueIndex] : nullptr;
        mhaDesc.OutputTensor = &outputDescs[outputIndex];
        mhaDesc.OutputPresentKeyTensor = hasPresentKeyOutput ? &outputDescs[outputPresentKeyIndex] : nullptr;
        mhaDesc.OutputPresentValueTensor = hasPresentValueOutput ? &outputDescs[outputPresentValueIndex] : nullptr;
        mhaDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(headSize)));
        mhaDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaDesc.HeadCount = numHeads;
        mhaDesc.MaskType = maskType;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &mhaDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MultiHeadAttention, DmlOperatorMultiHeadAttention);
} // namespace Dml
