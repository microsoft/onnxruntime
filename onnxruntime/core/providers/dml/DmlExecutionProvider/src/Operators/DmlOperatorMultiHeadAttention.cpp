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
    _Maybenull_ const DML_TENSOR_DESC* InputRelativePositionBiasTensor;
    const DML_TENSOR_DESC* OutputTensor;
    FLOAT MaskFilterValue;
    UINT NumHeads;
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

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == inputCount);

        // TODO (pavignol): Remove indices once we support additional inputs/outputs
        std::vector<std::optional<uint32_t>> inputIndices = {0, 1, 2, 3, 4, 5};
        std::vector<std::optional<uint32_t>> outputIndices = {0};
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices, std::nullopt, std::nullopt, 1);

        std::vector<uint32_t> queryTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(queryIndex);
        std::vector<uint32_t> keyTensorShape = m_inputTensorDescs[keyIndex].GetDmlDataType() == DML_TENSOR_TYPE_INVALID
            ? std::vector<uint32_t>()
            : kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(keyIndex);

        std::vector<uint32_t> valueTensorShape = m_inputTensorDescs[valueIndex].GetDmlDataType() == DML_TENSOR_TYPE_INVALID
            ? std::vector<uint32_t>()
            : kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(valueIndex);

        std::vector<uint32_t> biasTensorShape = m_inputTensorDescs[biasIndex].GetDmlDataType() == DML_TENSOR_TYPE_INVALID
            ? std::vector<uint32_t>()
            : kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(biasIndex);

        std::vector<uint32_t> relativePositionBiasTensorShape = m_inputTensorDescs[relativePositionBiasIndex].GetDmlDataType() == DML_TENSOR_TYPE_INVALID
            ? std::vector<uint32_t>()
            : kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(relativePositionBiasIndex);

        std::vector<uint32_t> pastKeyTensorShape;
        std::vector<uint32_t> pastValueTensorShape;

        // TODO (pavignol): Uncomment once we support additional inputs/outputs
        // std::vector<uint32_t> pastKeyTensorShape = m_inputTensorDescs[pastKeyIndex].GetDmlDataType() == DML_TENSOR_TYPE_INVALID
        //     ? std::vector<uint32_t>()
        //     : kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(pastKeyIndex);
        //
        // std::vector<uint32_t> pastValueTensorShape = m_inputTensorDescs[pastValueIndex].GetDmlDataType() == DML_TENSOR_TYPE_INVALID
        //     ? std::vector<uint32_t>()
        //     : kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(pastValueIndex);

        if (m_inputTensorDescs[keyIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 3);

            if (m_inputTensorDescs[valueIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
            {
                ML_CHECK_VALID_ARGUMENT(keyTensorShape.size() == 3);
                ML_CHECK_VALID_ARGUMENT(valueTensorShape.size() == 3);
            }
            else
            {
                ML_CHECK_VALID_ARGUMENT(keyTensorShape.size() == 5);
            }
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(queryTensorShape.size() == 5);
        }

        if (m_inputTensorDescs[biasIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(biasTensorShape.size() == 1);
        }

        if (m_inputTensorDescs[keyPaddingMaskIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            std::vector<uint32_t> keyPaddingMaskTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(keyPaddingMaskIndex);
            ML_CHECK_VALID_ARGUMENT(keyPaddingMaskTensorShape.size() == 1 || keyPaddingMaskTensorShape.size() == 2);

            if (keyPaddingMaskTensorShape.size() == 2)
            {
                const uint32_t batchSize = keyPaddingMaskTensorShape[0];
                const uint32_t kvSequenceLength = m_inputTensorDescs[keyIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID
                    ? keyTensorShape[1]
                    : queryTensorShape[1];

                const uint32_t broadcastedKeyPaddingMaskShape[2] = {batchSize, kvSequenceLength};

                // key_padding_mask's shape can be either [batch_size] or [batch_size, kv_sequence_length], so broadcast it if it's the former
                m_inputTensorDescs[keyPaddingMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(
                    m_inputTensorDescs[keyPaddingMaskIndex].GetMlOperatorDataType(),
                    broadcastedKeyPaddingMaskShape,
                    keyPaddingMaskTensorShape);
            }
        }

        if (m_inputTensorDescs[relativePositionBiasIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasTensorShape.size() == 4);
        }

        // if (m_inputTensorDescs[pastKeyIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        // {
        //     ML_CHECK_VALID_ARGUMENT(pastKeyTensorShape.size() == 4);
        // }
        //
        // if (m_inputTensorDescs[pastValueIndex].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        // {
        //     ML_CHECK_VALID_ARGUMENT(pastValueTensorShape.size() == 4);
        // }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MULTI_HEAD_ATTENTION_OPERATOR_DESC mhaDesc = {};
        mhaDesc.InputQueryTensor = &inputDescs[queryIndex];
        mhaDesc.InputKeyTensor = inputDescs[keyIndex].Desc ? &inputDescs[keyIndex] : nullptr;
        mhaDesc.InputValueTensor = inputDescs[valueIndex].Desc ? &inputDescs[valueIndex] : nullptr;
        mhaDesc.InputBiasTensor = inputDescs[biasIndex].Desc ? &inputDescs[biasIndex] : nullptr;
        mhaDesc.InputMaskTensor = inputDescs[keyPaddingMaskIndex].Desc ? &inputDescs[keyPaddingMaskIndex] : nullptr;
        mhaDesc.InputRelativePositionBiasTensor = inputDescs[relativePositionBiasIndex].Desc ? &inputDescs[relativePositionBiasIndex] : nullptr;
        mhaDesc.OutputTensor = &outputDescs[0];
        mhaDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaDesc.NumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        // TODO (pavignol): Support scale

        DML_OPERATOR_DESC opDesc = { static_cast<DML_OPERATOR_TYPE>(DML_INTERNAL_OPERATOR_MULTI_HEAD_ATTENTION), &mhaDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

// TODO (pavignol): Remove this once Past key/value are supported
void CALLBACK QueryMultiHeadAttention(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    if (context->IsInputValid(6) || context->IsInputValid(7))
    {
        *isSupported = false;
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(MultiHeadAttention, DmlOperatorMultiHeadAttention);
} // namespace Dml
