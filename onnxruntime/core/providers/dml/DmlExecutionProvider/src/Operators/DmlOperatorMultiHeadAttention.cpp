// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

// TODO (pavignol): Remove once we update to the latest DML version
struct DML_MULTI_HEAD_ATTENTION_OPERATOR_DESC
{
    const DML_TENSOR_DESC* InputQueryTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputValueTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputBiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputMaskTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputUnpaddedKeySequenceBoundsTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputRelativePositionBiasTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputPastKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* InputPastValueTensor;
    const DML_TENSOR_DESC* OutputTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputPresentKeyTensor;
    _Maybenull_ const DML_TENSOR_DESC* OutputPresentValueTensor;
    FLOAT MaskFilterValue;
    UINT NumHeads;
    FLOAT Scale;
    BOOL IsUnidirectional;
    BOOL DoRotary;
};

// TODO (pavignol): Remove once we update to the latest DML version
enum DML_INTERNAL_OPERATOR_TYPE
{
    DML_INTERNAL_OPERATOR_INVALID,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_IDENTITY,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ABS,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ACOS,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ADD,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ASIN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ATAN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_CEIL,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_CLIP,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_COS,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_DIVIDE,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_EXP,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_FLOOR,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOG,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_AND,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_NOT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_OR,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_XOR,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_MAX,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_MEAN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_MIN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_MULTIPLY,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_POW,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_CONSTANT_POW,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_RECIP,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_SIN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_SQRT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_SUBTRACT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_TAN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_THRESHOLD,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR,
    DML_INTERNAL_OPERATOR_ACTIVATION_ELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_HARDMAX,
    DML_INTERNAL_OPERATOR_ACTIVATION_HARD_SIGMOID,
    DML_INTERNAL_OPERATOR_ACTIVATION_IDENTITY,
    DML_INTERNAL_OPERATOR_ACTIVATION_LEAKY_RELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_LINEAR,
    DML_INTERNAL_OPERATOR_ACTIVATION_LOG_SOFTMAX,
    DML_INTERNAL_OPERATOR_ACTIVATION_PARAMETERIZED_RELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS,
    DML_INTERNAL_OPERATOR_ACTIVATION_RELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_SCALED_ELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_SCALED_TANH,
    DML_INTERNAL_OPERATOR_ACTIVATION_SIGMOID,
    DML_INTERNAL_OPERATOR_ACTIVATION_SOFTMAX,
    DML_INTERNAL_OPERATOR_ACTIVATION_SOFTPLUS,
    DML_INTERNAL_OPERATOR_ACTIVATION_SOFTSIGN,
    DML_INTERNAL_OPERATOR_ACTIVATION_TANH,
    DML_INTERNAL_OPERATOR_ACTIVATION_THRESHOLDED_RELU,
    DML_INTERNAL_OPERATOR_CONVOLUTION,
    DML_INTERNAL_OPERATOR_GEMM,
    DML_INTERNAL_OPERATOR_REDUCE,
    DML_INTERNAL_OPERATOR_AVERAGE_POOLING,
    DML_INTERNAL_OPERATOR_LP_POOLING,
    DML_INTERNAL_OPERATOR_MAX_POOLING,
    DML_INTERNAL_OPERATOR_ROI_POOLING,
    DML_INTERNAL_OPERATOR_SLICE,
    DML_INTERNAL_OPERATOR_CAST,
    DML_INTERNAL_OPERATOR_SPLIT,
    DML_INTERNAL_OPERATOR_JOIN,
    DML_INTERNAL_OPERATOR_PADDING,
    DML_INTERNAL_OPERATOR_VALUE_SCALE_2D,
    DML_INTERNAL_OPERATOR_UPSAMPLE_2D,
    DML_INTERNAL_OPERATOR_GATHER,
    DML_INTERNAL_OPERATOR_SPACE_TO_DEPTH,
    DML_INTERNAL_OPERATOR_DEPTH_TO_SPACE,
    DML_INTERNAL_OPERATOR_TILE,
    DML_INTERNAL_OPERATOR_TOP_K,
    DML_INTERNAL_OPERATOR_BATCH_NORMALIZATION,
    DML_INTERNAL_OPERATOR_MEAN_VARIANCE_NORMALIZATION,
    DML_INTERNAL_OPERATOR_LOCAL_RESPONSE_NORMALIZATION,
    DML_INTERNAL_OPERATOR_LP_NORMALIZATION,
    DML_INTERNAL_OPERATOR_RNN,
    DML_INTERNAL_OPERATOR_LSTM,
    DML_INTERNAL_OPERATOR_GRU,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_SIGN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_IS_NAN,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ERF,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_SINH,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_COSH,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_TANH,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ASINH,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ACOSH,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ATANH,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_IF,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ADD1,
    DML_INTERNAL_OPERATOR_ACTIVATION_SHRINK,
    DML_INTERNAL_OPERATOR_MAX_POOLING1,
    DML_INTERNAL_OPERATOR_MAX_UNPOOLING,
    DML_INTERNAL_OPERATOR_DIAGONAL_MATRIX,
    DML_INTERNAL_OPERATOR_SCATTER_ELEMENTS,
    DML_INTERNAL_OPERATOR_SCATTER = DML_OPERATOR_SCATTER_ELEMENTS, // Alias name for backwards compatibility.
    DML_INTERNAL_OPERATOR_ONE_HOT,
    DML_INTERNAL_OPERATOR_RESAMPLE,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ROUND,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_IS_INFINITY,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR,
    DML_INTERNAL_OPERATOR_FILL_VALUE_CONSTANT,
    DML_INTERNAL_OPERATOR_FILL_VALUE_SEQUENCE,
    DML_INTERNAL_OPERATOR_CUMULATIVE_SUMMATION,
    DML_INTERNAL_OPERATOR_REVERSE_SUBSEQUENCES,
    DML_INTERNAL_OPERATOR_GATHER_ELEMENTS,
    DML_INTERNAL_OPERATOR_GATHER_ND,
    DML_INTERNAL_OPERATOR_SCATTER_ND,
    DML_INTERNAL_OPERATOR_MAX_POOLING2,
    DML_INTERNAL_OPERATOR_SLICE1,
    DML_INTERNAL_OPERATOR_TOP_K1,
    DML_INTERNAL_OPERATOR_DEPTH_TO_SPACE1,
    DML_INTERNAL_OPERATOR_SPACE_TO_DEPTH1,
    DML_INTERNAL_OPERATOR_MEAN_VARIANCE_NORMALIZATION1,
    DML_INTERNAL_OPERATOR_RESAMPLE1,
    DML_INTERNAL_OPERATOR_MATRIX_MULTIPLY_INTEGER,
    DML_INTERNAL_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY,
    DML_INTERNAL_OPERATOR_CONVOLUTION_INTEGER,
    DML_INTERNAL_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_AND,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_OR,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_XOR,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_NOT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_BIT_COUNT,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL,
    DML_INTERNAL_OPERATOR_ACTIVATION_CELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_RELU_GRAD,
    DML_INTERNAL_OPERATOR_AVERAGE_POOLING_GRAD,
    DML_INTERNAL_OPERATOR_MAX_POOLING_GRAD,
    DML_INTERNAL_OPERATOR_RANDOM_GENERATOR,
    DML_INTERNAL_OPERATOR_NONZERO_COORDINATES,
    DML_INTERNAL_OPERATOR_RESAMPLE_GRAD,
    DML_INTERNAL_OPERATOR_SLICE_GRAD,
    DML_INTERNAL_OPERATOR_ADAM_OPTIMIZER,
    DML_INTERNAL_OPERATOR_ARGMIN,
    DML_INTERNAL_OPERATOR_ARGMAX,
    DML_INTERNAL_OPERATOR_ROI_ALIGN,
    DML_INTERNAL_OPERATOR_GATHER_ND1,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_ATAN_YX,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_CLIP_GRAD,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE,
    DML_INTERNAL_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD,
    DML_INTERNAL_OPERATOR_CUMULATIVE_PRODUCT,
    DML_INTERNAL_OPERATOR_BATCH_NORMALIZATION_GRAD,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD,
    DML_INTERNAL_OPERATOR_DYNAMIC_QUANTIZE_LINEAR,
    DML_INTERNAL_OPERATOR_ROI_ALIGN1,
    DML_INTERNAL_OPERATOR_ROI_ALIGN_GRAD,
    DML_INTERNAL_OPERATOR_BATCH_NORMALIZATION_TRAINING,
    DML_INTERNAL_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_CLIP1,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_CLIP_GRAD1,
    DML_INTERNAL_OPERATOR_PADDING1,
    DML_INTERNAL_OPERATOR_ELEMENT_WISE_NEGATE,
    DML_INTERNAL_OPERATOR_ACTIVATION_GELU,
    DML_INTERNAL_OPERATOR_ACTIVATION_SOFTMAX1,
    DML_INTERNAL_OPERATOR_ACTIVATION_LOG_SOFTMAX1,
    DML_INTERNAL_OPERATOR_ACTIVATION_HARDMAX1,
    DML_INTERNAL_OPERATOR_RESAMPLE2,
    DML_INTERNAL_OPERATOR_RESAMPLE_GRAD1,
    DML_INTERNAL_OPERATOR_DIAGONAL_MATRIX1,
    DML_INTERNAL_OPERATOR_MULTI_HEAD_ATTENTION,
};

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
            dmlUnpaddedKeySequenceBoundsIndex,
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

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == inputCount);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == outputCount);

        bool maskIsUnpaddedSequenceBounds =
            kernelCreationContext.IsInputValid(keyPaddingMaskIndex) &&
            kernelCreationContext.GetInputTensorDimensionCount(keyPaddingMaskIndex) == 1;

        bool keyValueIsPast =
            kernelCreationContext.IsInputValid(keyIndex) &&
            kernelCreationContext.GetInputTensorDimensionCount(keyIndex) == 4;

        std::vector<std::optional<uint32_t>> inputIndices = {
            queryIndex,
            keyValueIsPast ? std::nullopt : std::optional<uint32_t>(keyIndex),
            keyValueIsPast ? std::nullopt : std::optional<uint32_t>(valueIndex),
            biasIndex,
            maskIsUnpaddedSequenceBounds ? std::nullopt : std::optional<uint32_t>(keyPaddingMaskIndex),
            maskIsUnpaddedSequenceBounds ? std::optional<uint32_t>(keyPaddingMaskIndex) : std::nullopt,
            relativePositionBiasIndex,
            keyValueIsPast ? keyIndex : pastKeyIndex,
            keyValueIsPast ? valueIndex : pastValueIndex,
        };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, std::nullopt, std::nullopt, std::nullopt, 1);

        auto queryTensorShape = m_inputTensorDescs[dmlQueryIndex].GetSizes();
        ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3 || queryTensorShape.size() == 5);

        const uint32_t batchSize = queryTensorShape[0];
        const uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        const uint32_t headSize = queryTensorShape.size() == 5 ? queryTensorShape[4] : queryTensorShape[2] / numHeads;

        if (m_inputTensorDescs[dmlKeyIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3);

            if (m_inputTensorDescs[dmlValueIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 3 || m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 4);
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlValueIndex].GetDimensionCount() == m_inputTensorDescs[dmlKeyIndex].GetDimensionCount());
            }
            else
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlKeyIndex].GetDimensionCount() == 5);
            }
        }
        else if (m_inputTensorDescs[dmlPastKeyIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3);
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 5);
        }

        if (m_inputTensorDescs[dmlBiasIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlBiasIndex].GetDimensionCount() == 1);
        }

        if (m_inputTensorDescs[dmlKeyPaddingMaskIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
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

        if (m_inputTensorDescs[dmlUnpaddedKeySequenceBoundsIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            auto unpaddedKeySequenceBoundsShape = m_inputTensorDescs[dmlUnpaddedKeySequenceBoundsIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(unpaddedKeySequenceBoundsShape.size() == 1);
            ML_CHECK_VALID_ARGUMENT(unpaddedKeySequenceBoundsShape[0] % batchSize == 0);

            uint32_t desiredShape[2] = {unpaddedKeySequenceBoundsShape[0] / batchSize, batchSize};
            m_inputTensorDescs[dmlUnpaddedKeySequenceBoundsIndex] = TensorDesc(
                m_inputTensorDescs[dmlUnpaddedKeySequenceBoundsIndex].GetDmlDataType(),
                desiredShape);
        }

        if (m_inputTensorDescs[dmlRelativePositionBiasIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlRelativePositionBiasIndex].GetDimensionCount() == 4);
        }

        if (m_inputTensorDescs[dmlPastKeyIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastKeyIndex].GetDimensionCount() == 4);
        }

        if (m_inputTensorDescs[dmlPastValueIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[dmlPastValueIndex].GetDimensionCount() == 4);
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MULTI_HEAD_ATTENTION_OPERATOR_DESC mhaDesc = {};
        mhaDesc.InputQueryTensor = &inputDescs[dmlQueryIndex];
        mhaDesc.InputKeyTensor = inputDescs[dmlKeyIndex].Desc ? &inputDescs[dmlKeyIndex] : nullptr;
        mhaDesc.InputValueTensor = inputDescs[dmlValueIndex].Desc ? &inputDescs[dmlValueIndex] : nullptr;
        mhaDesc.InputBiasTensor = inputDescs[dmlBiasIndex].Desc ? &inputDescs[dmlBiasIndex] : nullptr;
        mhaDesc.InputMaskTensor = inputDescs[dmlKeyPaddingMaskIndex].Desc ? &inputDescs[dmlKeyPaddingMaskIndex] : nullptr;
        mhaDesc.InputUnpaddedKeySequenceBoundsTensor = inputDescs[dmlUnpaddedKeySequenceBoundsIndex].Desc ? &inputDescs[dmlUnpaddedKeySequenceBoundsIndex] : nullptr;
        mhaDesc.InputRelativePositionBiasTensor = inputDescs[dmlRelativePositionBiasIndex].Desc ? &inputDescs[dmlRelativePositionBiasIndex] : nullptr;
        mhaDesc.InputPastKeyTensor = inputDescs[dmlPastKeyIndex].Desc ? &inputDescs[dmlPastKeyIndex] : nullptr;
        mhaDesc.InputPastValueTensor = inputDescs[dmlPastValueIndex].Desc ? &inputDescs[dmlPastValueIndex] : nullptr;
        mhaDesc.OutputTensor = &outputDescs[outputIndex];
        mhaDesc.OutputPresentKeyTensor = outputDescs[outputPresentKeyIndex].Desc ? &outputDescs[outputPresentKeyIndex] : nullptr;
        mhaDesc.OutputPresentValueTensor = outputDescs[outputPresentValueIndex].Desc ? &outputDescs[outputPresentValueIndex] : nullptr;
        mhaDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaDesc.NumHeads = numHeads;
        mhaDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(headSize)));

        DML_OPERATOR_DESC opDesc = { static_cast<DML_OPERATOR_TYPE>(DML_INTERNAL_OPERATOR_MULTI_HEAD_ATTENTION), &mhaDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(MultiHeadAttention, DmlOperatorMultiHeadAttention);
} // namespace Dml
