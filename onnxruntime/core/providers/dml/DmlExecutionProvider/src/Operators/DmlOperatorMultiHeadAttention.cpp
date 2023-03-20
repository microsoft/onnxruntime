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
            keyPaddingMaskIndex,
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
            dmlBiasIndex,
            dmlKeyPaddingMaskIndex,
            dmlUnpaddedKeyBoundsIndex,
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
        const bool hasKey = kernelCreationContext.IsInputValid(keyIndex) && !keyValueIsPast;
        const bool hasValue = kernelCreationContext.IsInputValid(valueIndex) && !keyValueIsPast;
        const bool hasBias = kernelCreationContext.IsInputValid(biasIndex);
        const bool hasKeyPaddingMask = kernelCreationContext.IsInputValid(keyPaddingMaskIndex);
        const bool hasRelativePositionBias = kernelCreationContext.IsInputValid(relativePositionBiasIndex);
        const bool hasPastKey = keyValueIsPast || kernelCreationContext.IsInputValid(pastKeyIndex);
        const bool hasPastValue = keyValueIsPast || kernelCreationContext.IsInputValid(pastValueIndex);
        const bool hasPresentKeyOutput = kernelCreationContext.IsOutputValid(outputPresentKeyIndex);
        const bool hasPresentValueOutput = kernelCreationContext.IsOutputValid(outputPresentValueIndex);
        const bool maskIsUnpaddedBounds = hasKeyPaddingMask && kernelCreationContext.GetInputTensorDimensionCount(keyPaddingMaskIndex) == 1;

        std::vector<std::optional<uint32_t>> inputIndices = {
            queryIndex,
            keyValueIsPast ? std::nullopt : std::optional<uint32_t>(keyIndex),
            keyValueIsPast ? std::nullopt : std::optional<uint32_t>(valueIndex),
            biasIndex,
            maskIsUnpaddedBounds ? std::nullopt : std::optional<uint32_t>(keyPaddingMaskIndex),
            maskIsUnpaddedBounds ? std::optional<uint32_t>(keyPaddingMaskIndex) : std::nullopt,
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

        auto queryTensorShape = m_inputTensorDescs[dmlQueryIndex].GetSizes();
        ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3 || queryTensorShape.size() == 5);

        const uint32_t batchSize = queryTensorShape[0];
        const uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        const uint32_t headSize = queryTensorShape.size() == 5 ? queryTensorShape[4] : queryTensorShape[2] / numHeads;

        if (hasKey)
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3);

            if (hasValue)
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 3);
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetDimensionCount() == 3);
            }
            else
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 5);
            }
        }
        else if (hasPastKey)
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3);
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 5);
        }

        if (hasBias)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlBiasIndex].GetDimensionCount() == 1);
        }

        if (hasKeyPaddingMask && !maskIsUnpaddedBounds)
        {
            auto keyPaddingMaskTensorShape = m_inputTensorDescs[dmlKeyPaddingMaskIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(keyPaddingMaskTensorShape.size() == 2);

            const uint32_t kvSequenceLength = keyPaddingMaskTensorShape[1];
            const uint32_t sequenceLength = queryTensorShape[1];

            const uint32_t actualShape[4] = {batchSize, 1, 1, kvSequenceLength};
            const uint32_t desiredShape[4] = {batchSize, numHeads, sequenceLength, kvSequenceLength};

            m_inputTensorDescs[keyPaddingMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(
                m_inputTensorDescs[keyPaddingMaskIndex].GetMlOperatorDataType(),
                desiredShape,
                actualShape);
        }

        if (hasKeyPaddingMask && maskIsUnpaddedBounds)
        {
            auto unpaddedKeyBoundsShape = m_inputTensorDescs[dmlUnpaddedKeyBoundsIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape.size() == 1);
            ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape[0] % batchSize == 0);

            uint32_t desiredShape[2] = {unpaddedKeyBoundsShape[0] / batchSize, batchSize};
            m_inputTensorDescs[dmlUnpaddedKeyBoundsIndex] = TensorDesc(
                m_inputTensorDescs[dmlUnpaddedKeyBoundsIndex].GetDmlDataType(),
                desiredShape);
        }

        if (hasRelativePositionBias)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlRelativePositionBiasIndex].GetDimensionCount() == 4);
        }

        if (hasPastKey)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetDimensionCount() == 4);
        }

        if (hasPastValue)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetDimensionCount() == 4);
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MULTI_HEAD_ATTENTION_OPERATOR_DESC mhaDesc = {};
        mhaDesc.InputQueryTensor = &inputDescs[dmlQueryIndex];
        mhaDesc.InputKeyTensor = hasKey ? &inputDescs[dmlKeyIndex] : nullptr;
        mhaDesc.InputValueTensor = hasValue ? &inputDescs[dmlValueIndex] : nullptr;
        mhaDesc.InputBiasTensor = hasBias ? &inputDescs[dmlBiasIndex] : nullptr;
        mhaDesc.InputMaskTensor = (hasKeyPaddingMask && !maskIsUnpaddedBounds) ? &inputDescs[dmlKeyPaddingMaskIndex] : nullptr;
        mhaDesc.InputUnpaddedKeyBoundsTensor = (hasKeyPaddingMask && maskIsUnpaddedBounds) ? &inputDescs[dmlUnpaddedKeyBoundsIndex] : nullptr;
        mhaDesc.InputRelativePositionBiasTensor = hasRelativePositionBias ? &inputDescs[dmlRelativePositionBiasIndex] : nullptr;
        mhaDesc.InputPastKeyTensor = hasPastKey ? &inputDescs[dmlPastKeyIndex] : nullptr;
        mhaDesc.InputPastValueTensor = hasPastValue ? &inputDescs[dmlPastValueIndex] : nullptr;
        mhaDesc.OutputTensor = &outputDescs[outputIndex];
        mhaDesc.OutputPresentKeyTensor = hasPresentKeyOutput ? &outputDescs[outputPresentKeyIndex] : nullptr;
        mhaDesc.OutputPresentValueTensor = hasPresentValueOutput ? &outputDescs[outputPresentValueIndex] : nullptr;
        mhaDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaDesc.NumHeads = numHeads;
        mhaDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(headSize)));

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MULTI_HEAD_ATTENTION, &mhaDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MultiHeadAttention, DmlOperatorMultiHeadAttention);
} // namespace Dml
