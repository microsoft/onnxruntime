﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace ApiTraits
{
template <typename T>
struct EnumTraits
{
};

template <>
struct EnumTraits<DML_TENSOR_DATA_TYPE>
{
    static constexpr auto ValueCount = 14;
};

template <>
struct EnumTraits<DML_TENSOR_TYPE>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_OPERATOR_TYPE>
{
    static constexpr auto ValueCount = 174;
    static constexpr size_t ActivationFunctionCount = 26;
};

template <>
struct EnumTraits<DML_BINDING_TYPE>
{
    static constexpr auto ValueCount = 3;
};

template <>
struct EnumTraits<DML_REDUCE_FUNCTION>
{
    static constexpr auto ValueCount = 12;
    static constexpr DML_REDUCE_FUNCTION Invalid = static_cast<DML_REDUCE_FUNCTION>(ValueCount);
};

template <>
struct EnumTraits<DML_MATRIX_TRANSFORM>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_CONVOLUTION_MODE>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_CONVOLUTION_DIRECTION>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_PADDING_MODE>
{
    static constexpr auto ValueCount = 4;
};

template <>
struct EnumTraits<DML_INTERPOLATION_MODE>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_RECURRENT_NETWORK_DIRECTION>
{
    static constexpr auto ValueCount = 3;
};

template <>
struct EnumTraits<DML_FEATURE>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_FEATURE_LEVEL>
{
    static constexpr auto ValueCount = 14;
};

template <>
struct EnumTraits<DML_IS_INFINITY_MODE>
{
    static constexpr auto ValueCount = 3;
};

template <>
struct EnumTraits<DML_DEPTH_SPACE_ORDER>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_AXIS_DIRECTION>
{
    static constexpr auto ValueCount = 2;
};

template <>
struct EnumTraits<DML_ROUNDING_MODE>
{
    static constexpr auto ValueCount = 3;
};

template <>
struct EnumTraits<DML_RANDOM_GENERATOR_TYPE>
{
    static constexpr auto ValueCount = 1;
};

template <>
struct EnumTraits<DML_MULTIHEAD_ATTENTION_MASK_TYPE>
{
    static constexpr auto ValueCount = 5;
};

template <>
struct EnumTraits<DML_QUANTIZATION_TYPE>
{
    static constexpr auto ValueCount = 3;
};

template <typename T>
constexpr auto EnumValueCount = EnumTraits<T>::ValueCount;

template <typename T>
constexpr bool IsValidEnumValue(T value)
{
    return (std::make_unsigned_t<T>(value) < std::make_unsigned_t<T>(EnumValueCount<T>));
}

template <typename T>
struct FlagTraits
{
};

template <>
struct FlagTraits<DML_TENSOR_FLAGS>
{
    static constexpr auto ValidMask = DML_TENSOR_FLAG_OWNED_BY_DML;
};

template <>
struct FlagTraits<DML_EXECUTION_FLAGS>
{
    static constexpr auto ValidMask = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION | DML_EXECUTION_FLAG_DISABLE_META_COMMANDS | DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
};

template <>
struct FlagTraits<DML_CREATE_DEVICE_FLAGS>
{
    static constexpr auto ValidMask = DML_CREATE_DEVICE_FLAG_DEBUG;
};

template <typename T>
constexpr auto FlagsValidMask = FlagTraits<T>::ValidMask;

template <typename T>
constexpr bool IsValidFlags(T value)
{
    return (value & ~FlagsValidMask<T>) == 0;
}

template <typename T>
struct TensorDescTraits
{
};

template <>
struct TensorDescTraits<DML_BUFFER_TENSOR_DESC>
{
    static constexpr DML_TENSOR_TYPE Type = DML_TENSOR_TYPE_BUFFER;
};


template <typename T>
struct OperatorDescTraits
{
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ABS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ABS;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ACOS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ACOS;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ADD;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ASIN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ASIN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ATAN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ATAN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_CEIL_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_CEIL;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_CLIP_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_CLIP;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_CLIP1;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_COS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_COS;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_DIVIDE;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_EXP_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_EXP;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_FLOOR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOG_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOG;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_MAX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_MAX;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_MEAN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_MEAN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_MIN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_MIN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_POW_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_POW;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_RECIP;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_SIN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_SIN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_SQRT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_SQRT;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ATAN_YX;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_SUBTRACT;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_TAN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_TAN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_THRESHOLD;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR;
};

template <>
struct OperatorDescTraits<DML_CONVOLUTION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_CONVOLUTION;
};

template <>
struct OperatorDescTraits<DML_GEMM_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_GEMM;
};

template <>
struct OperatorDescTraits<DML_REDUCE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_REDUCE;
};

template <>
struct OperatorDescTraits<DML_ARGMIN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ARGMIN;
};

template <>
struct OperatorDescTraits<DML_ARGMAX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ARGMAX;
};

template <>
struct OperatorDescTraits<DML_AVERAGE_POOLING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_AVERAGE_POOLING;
};

template <>
struct OperatorDescTraits<DML_AVERAGE_POOLING1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_AVERAGE_POOLING1;
};

template <>
struct OperatorDescTraits<DML_LP_POOLING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_LP_POOLING;
};

template <>
struct OperatorDescTraits<DML_LP_POOLING1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_LP_POOLING1;
};

template <>
struct OperatorDescTraits<DML_MAX_POOLING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MAX_POOLING;
};

template <>
struct OperatorDescTraits<DML_MAX_POOLING1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MAX_POOLING1;
};

template <>
struct OperatorDescTraits<DML_ROI_POOLING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ROI_POOLING;
};

template <>
struct OperatorDescTraits<DML_SLICE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SLICE;
};

template <>
struct OperatorDescTraits<DML_CAST_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_CAST;
};

template <>
struct OperatorDescTraits<DML_SPLIT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SPLIT;
};

template <>
struct OperatorDescTraits<DML_JOIN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_JOIN;
};

template <>
struct OperatorDescTraits<DML_PADDING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_PADDING;
};

template <>
struct OperatorDescTraits<DML_PADDING1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_PADDING1;
};

template <>
struct OperatorDescTraits<DML_VALUE_SCALE_2D_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_VALUE_SCALE_2D;
};

template <>
struct OperatorDescTraits<DML_UPSAMPLE_2D_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_UPSAMPLE_2D;
};

template <>
struct OperatorDescTraits<DML_GATHER_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_GATHER;
};

template <>
struct OperatorDescTraits<DML_SPACE_TO_DEPTH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SPACE_TO_DEPTH;
};

template <>
struct OperatorDescTraits<DML_DEPTH_TO_SPACE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_DEPTH_TO_SPACE;
};

template <>
struct OperatorDescTraits<DML_TILE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_TILE;
};

template <>
struct OperatorDescTraits<DML_TOP_K_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_TOP_K;
};

template <>
struct OperatorDescTraits<DML_BATCH_NORMALIZATION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_BATCH_NORMALIZATION;
};

template <>
struct OperatorDescTraits<DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_BATCH_NORMALIZATION_GRAD;
};

template <>
struct OperatorDescTraits<DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD;
};

template <>
struct OperatorDescTraits<DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION;
};

template <>
struct OperatorDescTraits<DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION;
};

template <>
struct OperatorDescTraits<DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD;
};

template <>
struct OperatorDescTraits<DML_LP_NORMALIZATION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_LP_NORMALIZATION;
};

template <>
struct OperatorDescTraits<DML_RNN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RNN;
};

template <>
struct OperatorDescTraits<DML_LSTM_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_LSTM;
};

template <>
struct OperatorDescTraits<DML_GRU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_GRU;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_SIGN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_SIGN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_IS_NAN;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_NEGATE;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ERF_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ERF;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_SINH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_SINH;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_COSH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_COSH;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_TANH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_TANH;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ASINH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ASINH;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ACOSH;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ATANH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ATANH;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_IF_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_IF;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ADD1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ADD1;
};

template <>
struct OperatorDescTraits<DML_MAX_UNPOOLING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MAX_UNPOOLING;
};

template <>
struct OperatorDescTraits<DML_DIAGONAL_MATRIX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_DIAGONAL_MATRIX;
};

template <>
struct OperatorDescTraits<DML_SCATTER_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SCATTER;
};

template <>
struct OperatorDescTraits<DML_ONE_HOT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ONE_HOT;
};

template <>
struct OperatorDescTraits<DML_RESAMPLE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RESAMPLE;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_ROUND_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_ROUND;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_IS_INFINITY;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR;
};

template <>
struct OperatorDescTraits<DML_FILL_VALUE_CONSTANT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
};

template <>
struct OperatorDescTraits<DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_FILL_VALUE_SEQUENCE;
};

template <>
struct OperatorDescTraits<DML_CUMULATIVE_SUMMATION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_CUMULATIVE_SUMMATION;
};

template <>
struct OperatorDescTraits<DML_CUMULATIVE_PRODUCT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_CUMULATIVE_PRODUCT;
};

template <>
struct OperatorDescTraits<DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_REVERSE_SUBSEQUENCES;
};

template <>
struct OperatorDescTraits<DML_GATHER_ELEMENTS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_GATHER_ELEMENTS;
};

template <>
struct OperatorDescTraits<DML_GATHER_ND_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_GATHER_ND;
};

template <>
struct OperatorDescTraits<DML_SCATTER_ND_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SCATTER_ND;
};

template <>
struct OperatorDescTraits<DML_MAX_POOLING2_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MAX_POOLING2;
};

template <>
struct OperatorDescTraits<DML_SLICE1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SLICE1;
};

template <>
struct OperatorDescTraits<DML_TOP_K1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_TOP_K1;
};

template <>
struct OperatorDescTraits<DML_DEPTH_TO_SPACE1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_DEPTH_TO_SPACE1;
};

template <>
struct OperatorDescTraits<DML_SPACE_TO_DEPTH1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SPACE_TO_DEPTH1;
};

template <>
struct OperatorDescTraits<DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1;
};

template <>
struct OperatorDescTraits<DML_RESAMPLE1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RESAMPLE1;
};

template <>
struct OperatorDescTraits<DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MATRIX_MULTIPLY_INTEGER;
};

template <>
struct OperatorDescTraits<DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY;
};

template <>
struct OperatorDescTraits<DML_CONVOLUTION_INTEGER_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_CONVOLUTION_INTEGER;
};

template <>
struct OperatorDescTraits<DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_AND;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_OR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_XOR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_NOT;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_BIT_COUNT;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_RELU_GRAD;
};

template <>
struct OperatorDescTraits<DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_AVERAGE_POOLING_GRAD;
};

template <>
struct OperatorDescTraits<DML_MAX_POOLING_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MAX_POOLING_GRAD;
};

template <>
struct OperatorDescTraits<DML_RANDOM_GENERATOR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RANDOM_GENERATOR;
};

template <>
struct OperatorDescTraits<DML_NONZERO_COORDINATES_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_NONZERO_COORDINATES;
};

template <>
struct OperatorDescTraits<DML_RESAMPLE_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RESAMPLE_GRAD;
};

template <>
struct OperatorDescTraits<DML_SLICE_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_SLICE_GRAD;
};

template <>
struct OperatorDescTraits<DML_ADAM_OPTIMIZER_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ADAM_OPTIMIZER;
};

template <>
struct OperatorDescTraits<DML_ROI_ALIGN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ROI_ALIGN;
};

template <>
struct OperatorDescTraits<DML_ROI_ALIGN1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ROI_ALIGN1;
};

template <>
struct OperatorDescTraits<DML_GATHER_ND1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_GATHER_ND1;
};

template <>
struct OperatorDescTraits<DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR;
};

template <>
struct OperatorDescTraits<DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD;
};

template <>
struct OperatorDescTraits<DML_ROI_ALIGN_GRAD_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ROI_ALIGN_GRAD;
};

template <>
struct OperatorDescTraits<DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_BATCH_NORMALIZATION_TRAINING;
};

template <>
struct OperatorDescTraits<DML_RESAMPLE2_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RESAMPLE2;
};

template <>
struct OperatorDescTraits<DML_RESAMPLE_GRAD1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_RESAMPLE_GRAD1;
};

template <>
struct OperatorDescTraits<DML_DIAGONAL_MATRIX1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_DIAGONAL_MATRIX1;
};

template <>
struct OperatorDescTraits<DML_MULTIHEAD_ATTENTION_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MULTIHEAD_ATTENTION;
};

template <>
struct OperatorDescTraits<DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING;
};

template <>
struct OperatorDescTraits<DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT;
};

template <>
struct OperatorDescTraits<DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2;
};

template <>
struct OperatorDescTraits<DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_MULTIHEAD_ATTENTION1;
};

template <>
struct OperatorDescTraits<DML_QUANTIZE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_QUANTIZE;
};

template <>
struct OperatorDescTraits<DML_DEQUANTIZE_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_DEQUANTIZE;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_ELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_ELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_CELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_CELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_HARDMAX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_HARDMAX;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_HARDMAX1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_HARDMAX1;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_HARD_SIGMOID;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_IDENTITY_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_IDENTITY;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_LEAKY_RELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_LINEAR_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_LINEAR;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_RELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_RELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SCALED_ELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SCALED_TANH;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SIGMOID;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SOFTMAX_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SOFTMAX;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SOFTPLUS;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SOFTSIGN;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_TANH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_TANH;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SHRINK_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SHRINK;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_GELU_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_GELU;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_SWISH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_SWISH;
};

template <>
struct OperatorDescTraits<DML_ACTIVATION_HARD_SWISH_OPERATOR_DESC>
{
    static constexpr DML_OPERATOR_TYPE Type = DML_OPERATOR_ACTIVATION_HARD_SWISH;
};


template <DML_OPERATOR_TYPE Type>
struct OperatorTypeTraits
{
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_IDENTITY>
{
    using DescType = DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ABS>
{
    using DescType = DML_ELEMENT_WISE_ABS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ACOS>
{
    using DescType = DML_ELEMENT_WISE_ACOS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ADD>
{
    using DescType = DML_ELEMENT_WISE_ADD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ASIN>
{
    using DescType = DML_ELEMENT_WISE_ASIN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ATAN>
{
    using DescType = DML_ELEMENT_WISE_ATAN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_CEIL>
{
    using DescType = DML_ELEMENT_WISE_CEIL_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_CLIP>
{
    using DescType = DML_ELEMENT_WISE_CLIP_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_CLIP1>
{
    using DescType = DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD>
{
    using DescType = DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1>
{
    using DescType = DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_COS>
{
    using DescType = DML_ELEMENT_WISE_COS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_DIVIDE>
{
    using DescType = DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_EXP>
{
    using DescType = DML_ELEMENT_WISE_EXP_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_FLOOR>
{
    using DescType = DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOG>
{
    using DescType = DML_ELEMENT_WISE_LOG_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR>
{
    using DescType = DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_MAX>
{
    using DescType = DML_ELEMENT_WISE_MAX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_MEAN>
{
    using DescType = DML_ELEMENT_WISE_MEAN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_MIN>
{
    using DescType = DML_ELEMENT_WISE_MIN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_MULTIPLY>
{
    using DescType = DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_POW>
{
    using DescType = DML_ELEMENT_WISE_POW_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW>
{
    using DescType = DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_RECIP>
{
    using DescType = DML_ELEMENT_WISE_RECIP_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_SIN>
{
    using DescType = DML_ELEMENT_WISE_SIN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_SQRT>
{
    using DescType = DML_ELEMENT_WISE_SQRT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE>
{
    using DescType = DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ATAN_YX>
{
    using DescType = DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_SUBTRACT>
{
    using DescType = DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_TAN>
{
    using DescType = DML_ELEMENT_WISE_TAN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_THRESHOLD>
{
    using DescType = DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR>
{
    using DescType = DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR>
{
    using DescType = DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_CONVOLUTION>
{
    using DescType = DML_CONVOLUTION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_GEMM>
{
    using DescType = DML_GEMM_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_REDUCE>
{
    using DescType = DML_REDUCE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ARGMIN>
{
    using DescType = DML_ARGMIN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ARGMAX>
{
    using DescType = DML_ARGMAX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_AVERAGE_POOLING>
{
    using DescType = DML_AVERAGE_POOLING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_AVERAGE_POOLING1>
{
    using DescType = DML_AVERAGE_POOLING1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_LP_POOLING>
{
    using DescType = DML_LP_POOLING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_LP_POOLING1>
{
    using DescType = DML_LP_POOLING1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MAX_POOLING>
{
    using DescType = DML_MAX_POOLING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MAX_POOLING1>
{
    using DescType = DML_MAX_POOLING1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ROI_POOLING>
{
    using DescType = DML_ROI_POOLING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SLICE>
{
    using DescType = DML_SLICE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_CAST>
{
    using DescType = DML_CAST_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SPLIT>
{
    using DescType = DML_SPLIT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_JOIN>
{
    using DescType = DML_JOIN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_PADDING>
{
    using DescType = DML_PADDING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_PADDING1>
{
    using DescType = DML_PADDING1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_VALUE_SCALE_2D>
{
    using DescType = DML_VALUE_SCALE_2D_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_UPSAMPLE_2D>
{
    using DescType = DML_UPSAMPLE_2D_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_GATHER>
{
    using DescType = DML_GATHER_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SPACE_TO_DEPTH>
{
    using DescType = DML_SPACE_TO_DEPTH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_DEPTH_TO_SPACE>
{
    using DescType = DML_DEPTH_TO_SPACE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_TILE>
{
    using DescType = DML_TILE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_TOP_K>
{
    using DescType = DML_TOP_K_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_BATCH_NORMALIZATION>
{
    using DescType = DML_BATCH_NORMALIZATION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_BATCH_NORMALIZATION_GRAD>
{
    using DescType = DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD>
{
    using DescType = DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION>
{
    using DescType = DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION>
{
    using DescType = DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD>
{
    using DescType = DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_LP_NORMALIZATION>
{
    using DescType = DML_LP_NORMALIZATION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RNN>
{
    using DescType = DML_RNN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_LSTM>
{
    using DescType = DML_LSTM_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_GRU>
{
    using DescType = DML_GRU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_SIGN>
{
    using DescType = DML_ELEMENT_WISE_SIGN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_IS_NAN>
{
    using DescType = DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_NEGATE>
{
    using DescType = DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ERF>
{
    using DescType = DML_ELEMENT_WISE_ERF_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_SINH>
{
    using DescType = DML_ELEMENT_WISE_SINH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_COSH>
{
    using DescType = DML_ELEMENT_WISE_COSH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_TANH>
{
    using DescType = DML_ELEMENT_WISE_TANH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ASINH>
{
    using DescType = DML_ELEMENT_WISE_ASINH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ACOSH>
{
    using DescType = DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ATANH>
{
    using DescType = DML_ELEMENT_WISE_ATANH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_IF>
{
    using DescType = DML_ELEMENT_WISE_IF_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ADD1>
{
    using DescType = DML_ELEMENT_WISE_ADD1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MAX_UNPOOLING>
{
    using DescType = DML_MAX_UNPOOLING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_DIAGONAL_MATRIX>
{
    using DescType = DML_DIAGONAL_MATRIX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SCATTER>
{
    using DescType = DML_SCATTER_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ONE_HOT>
{
    using DescType = DML_ONE_HOT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RESAMPLE>
{
    using DescType = DML_RESAMPLE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT>
{
    using DescType = DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT>
{
    using DescType = DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_ROUND>
{
    using DescType = DML_ELEMENT_WISE_ROUND_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_IS_INFINITY>
{
    using DescType = DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE>
{
    using DescType = DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR>
{
    using DescType = DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_FILL_VALUE_CONSTANT>
{
    using DescType = DML_FILL_VALUE_CONSTANT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_FILL_VALUE_SEQUENCE>
{
    using DescType = DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_CUMULATIVE_SUMMATION>
{
    using DescType = DML_CUMULATIVE_SUMMATION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_CUMULATIVE_PRODUCT>
{
    using DescType = DML_CUMULATIVE_PRODUCT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_REVERSE_SUBSEQUENCES>
{
    using DescType = DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_GATHER_ELEMENTS>
{
    using DescType = DML_GATHER_ELEMENTS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_GATHER_ND>
{
    using DescType = DML_GATHER_ND_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SCATTER_ND>
{
    using DescType = DML_SCATTER_ND_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MAX_POOLING2>
{
    using DescType = DML_MAX_POOLING2_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SLICE1>
{
    using DescType = DML_SLICE1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_TOP_K1>
{
    using DescType = DML_TOP_K1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_DEPTH_TO_SPACE1>
{
    using DescType = DML_DEPTH_TO_SPACE1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SPACE_TO_DEPTH1>
{
    using DescType = DML_SPACE_TO_DEPTH1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1>
{
    using DescType = DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RESAMPLE1>
{
    using DescType = DML_RESAMPLE1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MATRIX_MULTIPLY_INTEGER>
{
    using DescType = DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY>
{
    using DescType = DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_CONVOLUTION_INTEGER>
{
    using DescType = DML_CONVOLUTION_INTEGER_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION>
{
    using DescType = DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_AND>
{
    using DescType = DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_OR>
{
    using DescType = DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_XOR>
{
    using DescType = DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_NOT>
{
    using DescType = DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_BIT_COUNT>
{
    using DescType = DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_RELU_GRAD>
{
    using DescType = DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_AVERAGE_POOLING_GRAD>
{
    using DescType = DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MAX_POOLING_GRAD>
{
    using DescType = DML_MAX_POOLING_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RANDOM_GENERATOR>
{
    using DescType = DML_RANDOM_GENERATOR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_NONZERO_COORDINATES>
{
    using DescType = DML_NONZERO_COORDINATES_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RESAMPLE_GRAD>
{
    using DescType = DML_RESAMPLE_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_SLICE_GRAD>
{
    using DescType = DML_SLICE_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ADAM_OPTIMIZER>
{
    using DescType = DML_ADAM_OPTIMIZER_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ROI_ALIGN>
{
    using DescType = DML_ROI_ALIGN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ROI_ALIGN1>
{
    using DescType = DML_ROI_ALIGN1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_GATHER_ND1>
{
    using DescType = DML_GATHER_ND1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR>
{
    using DescType = DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD>
{
    using DescType = DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ROI_ALIGN_GRAD>
{
    using DescType = DML_ROI_ALIGN_GRAD_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_BATCH_NORMALIZATION_TRAINING>
{
    using DescType = DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RESAMPLE2>
{
    using DescType = DML_RESAMPLE2_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_RESAMPLE_GRAD1>
{
    using DescType = DML_RESAMPLE_GRAD1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_DIAGONAL_MATRIX1>
{
    using DescType = DML_DIAGONAL_MATRIX1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MULTIHEAD_ATTENTION>
{
    using DescType = DML_MULTIHEAD_ATTENTION_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING>
{
    using DescType = DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT>
{
    using DescType = DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2>
{
    using DescType = DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_MULTIHEAD_ATTENTION1>
{
    using DescType = DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_QUANTIZE>
{
    using DescType = DML_QUANTIZE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_DEQUANTIZE>
{
    using DescType = DML_DEQUANTIZE_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_ELU>
{
    using DescType = DML_ACTIVATION_ELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_CELU>
{
    using DescType = DML_ACTIVATION_CELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_HARDMAX>
{
    using DescType = DML_ACTIVATION_HARDMAX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_HARDMAX1>
{
    using DescType = DML_ACTIVATION_HARDMAX1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_HARD_SIGMOID>
{
    using DescType = DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_IDENTITY>
{
    using DescType = DML_ACTIVATION_IDENTITY_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_LEAKY_RELU>
{
    using DescType = DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_LINEAR>
{
    using DescType = DML_ACTIVATION_LINEAR_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_LOG_SOFTMAX>
{
    using DescType = DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1>
{
    using DescType = DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU>
{
    using DescType = DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS>
{
    using DescType = DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_RELU>
{
    using DescType = DML_ACTIVATION_RELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SCALED_ELU>
{
    using DescType = DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SCALED_TANH>
{
    using DescType = DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SIGMOID>
{
    using DescType = DML_ACTIVATION_SIGMOID_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SOFTMAX>
{
    using DescType = DML_ACTIVATION_SOFTMAX_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SOFTMAX1>
{
    using DescType = DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SOFTPLUS>
{
    using DescType = DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SOFTSIGN>
{
    using DescType = DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_TANH>
{
    using DescType = DML_ACTIVATION_TANH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU>
{
    using DescType = DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SHRINK>
{
    using DescType = DML_ACTIVATION_SHRINK_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_GELU>
{
    using DescType = DML_ACTIVATION_GELU_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_SWISH>
{
    using DescType = DML_ACTIVATION_SWISH_OPERATOR_DESC;
};

template <>
struct OperatorTypeTraits<(DML_OPERATOR_TYPE)DML_OPERATOR_ACTIVATION_HARD_SWISH>
{
    using DescType = DML_ACTIVATION_HARD_SWISH_OPERATOR_DESC;
};


// Calls a visitor functor, supplying an empty operator desc corresponding to the given DML_OPERATOR_TYPE as
// the first argument.
//
// For example:
//   Visit(DML_OPERATOR_ELEMENT_WISE_IDENTITY, [](auto tag) {
//       using T = decltype(tag); // T is one of the DML_*_OPERATOR_DESC structs
//   });
//
template <typename Visitor, typename... Ts>
auto OperatorTypeVisitor(DML_OPERATOR_TYPE type, Visitor&& visitor, Ts&&... args)
{
    switch (static_cast<uint32_t>(type))
    {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ABS:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ABS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ACOS:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ACOS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ADD:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ADD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ASIN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ASIN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ATAN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ATAN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_CEIL:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_CEIL_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_CLIP:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_CLIP_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_CLIP1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_COS:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_COS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_EXP:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_EXP_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_FLOOR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOG:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOG_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_MAX:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_MAX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_MEAN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_MEAN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_MIN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_MIN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_POW:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_POW_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_RECIP:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_RECIP_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_SIN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_SIN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_SQRT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_SQRT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ATAN_YX:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_TAN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_TAN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_THRESHOLD:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_CONVOLUTION:
        return std::invoke(std::forward<Visitor>(visitor), DML_CONVOLUTION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_GEMM:
        return std::invoke(std::forward<Visitor>(visitor), DML_GEMM_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_REDUCE:
        return std::invoke(std::forward<Visitor>(visitor), DML_REDUCE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ARGMIN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ARGMIN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ARGMAX:
        return std::invoke(std::forward<Visitor>(visitor), DML_ARGMAX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_AVERAGE_POOLING:
        return std::invoke(std::forward<Visitor>(visitor), DML_AVERAGE_POOLING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_AVERAGE_POOLING1:
        return std::invoke(std::forward<Visitor>(visitor), DML_AVERAGE_POOLING1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_LP_POOLING:
        return std::invoke(std::forward<Visitor>(visitor), DML_LP_POOLING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_LP_POOLING1:
        return std::invoke(std::forward<Visitor>(visitor), DML_LP_POOLING1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MAX_POOLING:
        return std::invoke(std::forward<Visitor>(visitor), DML_MAX_POOLING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MAX_POOLING1:
        return std::invoke(std::forward<Visitor>(visitor), DML_MAX_POOLING1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ROI_POOLING:
        return std::invoke(std::forward<Visitor>(visitor), DML_ROI_POOLING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SLICE:
        return std::invoke(std::forward<Visitor>(visitor), DML_SLICE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_CAST:
        return std::invoke(std::forward<Visitor>(visitor), DML_CAST_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SPLIT:
        return std::invoke(std::forward<Visitor>(visitor), DML_SPLIT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_JOIN:
        return std::invoke(std::forward<Visitor>(visitor), DML_JOIN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_PADDING:
        return std::invoke(std::forward<Visitor>(visitor), DML_PADDING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_PADDING1:
        return std::invoke(std::forward<Visitor>(visitor), DML_PADDING1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_VALUE_SCALE_2D:
        return std::invoke(std::forward<Visitor>(visitor), DML_VALUE_SCALE_2D_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_UPSAMPLE_2D:
        return std::invoke(std::forward<Visitor>(visitor), DML_UPSAMPLE_2D_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_GATHER:
        return std::invoke(std::forward<Visitor>(visitor), DML_GATHER_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SPACE_TO_DEPTH:
        return std::invoke(std::forward<Visitor>(visitor), DML_SPACE_TO_DEPTH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_DEPTH_TO_SPACE:
        return std::invoke(std::forward<Visitor>(visitor), DML_DEPTH_TO_SPACE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_TILE:
        return std::invoke(std::forward<Visitor>(visitor), DML_TILE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_TOP_K:
        return std::invoke(std::forward<Visitor>(visitor), DML_TOP_K_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_BATCH_NORMALIZATION:
        return std::invoke(std::forward<Visitor>(visitor), DML_BATCH_NORMALIZATION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_BATCH_NORMALIZATION_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION:
        return std::invoke(std::forward<Visitor>(visitor), DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION:
        return std::invoke(std::forward<Visitor>(visitor), DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_LP_NORMALIZATION:
        return std::invoke(std::forward<Visitor>(visitor), DML_LP_NORMALIZATION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RNN:
        return std::invoke(std::forward<Visitor>(visitor), DML_RNN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_LSTM:
        return std::invoke(std::forward<Visitor>(visitor), DML_LSTM_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_GRU:
        return std::invoke(std::forward<Visitor>(visitor), DML_GRU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_SIGN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_SIGN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_IS_NAN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_NEGATE:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ERF:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ERF_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_SINH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_SINH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_COSH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_COSH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_TANH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_TANH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ASINH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ASINH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ACOSH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ATANH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ATANH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_IF:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_IF_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ADD1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ADD1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MAX_UNPOOLING:
        return std::invoke(std::forward<Visitor>(visitor), DML_MAX_UNPOOLING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_DIAGONAL_MATRIX:
        return std::invoke(std::forward<Visitor>(visitor), DML_DIAGONAL_MATRIX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SCATTER:
        return std::invoke(std::forward<Visitor>(visitor), DML_SCATTER_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ONE_HOT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ONE_HOT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RESAMPLE:
        return std::invoke(std::forward<Visitor>(visitor), DML_RESAMPLE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_ROUND:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_ROUND_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_IS_INFINITY:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_FILL_VALUE_CONSTANT:
        return std::invoke(std::forward<Visitor>(visitor), DML_FILL_VALUE_CONSTANT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_FILL_VALUE_SEQUENCE:
        return std::invoke(std::forward<Visitor>(visitor), DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_CUMULATIVE_SUMMATION:
        return std::invoke(std::forward<Visitor>(visitor), DML_CUMULATIVE_SUMMATION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_CUMULATIVE_PRODUCT:
        return std::invoke(std::forward<Visitor>(visitor), DML_CUMULATIVE_PRODUCT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_REVERSE_SUBSEQUENCES:
        return std::invoke(std::forward<Visitor>(visitor), DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_GATHER_ELEMENTS:
        return std::invoke(std::forward<Visitor>(visitor), DML_GATHER_ELEMENTS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_GATHER_ND:
        return std::invoke(std::forward<Visitor>(visitor), DML_GATHER_ND_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SCATTER_ND:
        return std::invoke(std::forward<Visitor>(visitor), DML_SCATTER_ND_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MAX_POOLING2:
        return std::invoke(std::forward<Visitor>(visitor), DML_MAX_POOLING2_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SLICE1:
        return std::invoke(std::forward<Visitor>(visitor), DML_SLICE1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_TOP_K1:
        return std::invoke(std::forward<Visitor>(visitor), DML_TOP_K1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_DEPTH_TO_SPACE1:
        return std::invoke(std::forward<Visitor>(visitor), DML_DEPTH_TO_SPACE1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SPACE_TO_DEPTH1:
        return std::invoke(std::forward<Visitor>(visitor), DML_SPACE_TO_DEPTH1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1:
        return std::invoke(std::forward<Visitor>(visitor), DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RESAMPLE1:
        return std::invoke(std::forward<Visitor>(visitor), DML_RESAMPLE1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER:
        return std::invoke(std::forward<Visitor>(visitor), DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY:
        return std::invoke(std::forward<Visitor>(visitor), DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_CONVOLUTION_INTEGER:
        return std::invoke(std::forward<Visitor>(visitor), DML_CONVOLUTION_INTEGER_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION:
        return std::invoke(std::forward<Visitor>(visitor), DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_AND:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_OR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_XOR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_NOT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_BIT_COUNT:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_RELU_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_AVERAGE_POOLING_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MAX_POOLING_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_MAX_POOLING_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RANDOM_GENERATOR:
        return std::invoke(std::forward<Visitor>(visitor), DML_RANDOM_GENERATOR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_NONZERO_COORDINATES:
        return std::invoke(std::forward<Visitor>(visitor), DML_NONZERO_COORDINATES_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RESAMPLE_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_RESAMPLE_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_SLICE_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_SLICE_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ADAM_OPTIMIZER:
        return std::invoke(std::forward<Visitor>(visitor), DML_ADAM_OPTIMIZER_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ROI_ALIGN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ROI_ALIGN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ROI_ALIGN1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ROI_ALIGN1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_GATHER_ND1:
        return std::invoke(std::forward<Visitor>(visitor), DML_GATHER_ND1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR:
        return std::invoke(std::forward<Visitor>(visitor), DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD:
        return std::invoke(std::forward<Visitor>(visitor), DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ROI_ALIGN_GRAD:
        return std::invoke(std::forward<Visitor>(visitor), DML_ROI_ALIGN_GRAD_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING:
        return std::invoke(std::forward<Visitor>(visitor), DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RESAMPLE2:
        return std::invoke(std::forward<Visitor>(visitor), DML_RESAMPLE2_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_RESAMPLE_GRAD1:
        return std::invoke(std::forward<Visitor>(visitor), DML_RESAMPLE_GRAD1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_DIAGONAL_MATRIX1:
        return std::invoke(std::forward<Visitor>(visitor), DML_DIAGONAL_MATRIX1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MULTIHEAD_ATTENTION:
        return std::invoke(std::forward<Visitor>(visitor), DML_MULTIHEAD_ATTENTION_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING:
        return std::invoke(std::forward<Visitor>(visitor), DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT:
        return std::invoke(std::forward<Visitor>(visitor), DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2:
        return std::invoke(std::forward<Visitor>(visitor), DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_MULTIHEAD_ATTENTION1:
        return std::invoke(std::forward<Visitor>(visitor), DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_QUANTIZE:
        return std::invoke(std::forward<Visitor>(visitor), DML_QUANTIZE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_DEQUANTIZE:
        return std::invoke(std::forward<Visitor>(visitor), DML_DEQUANTIZE_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_ELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_ELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_CELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_CELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_HARDMAX:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_HARDMAX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_HARDMAX1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_HARDMAX1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_HARD_SIGMOID:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_IDENTITY:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_IDENTITY_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_LEAKY_RELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_LINEAR:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_LINEAR_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_RELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_RELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SCALED_ELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SCALED_TANH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SIGMOID:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SIGMOID_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SOFTMAX:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SOFTMAX_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SOFTMAX1:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SOFTPLUS:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SOFTSIGN:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_TANH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_TANH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SHRINK:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SHRINK_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_GELU:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_GELU_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_SWISH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_SWISH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    case DML_OPERATOR_ACTIVATION_HARD_SWISH:
        return std::invoke(std::forward<Visitor>(visitor), DML_ACTIVATION_HARD_SWISH_OPERATOR_DESC{}, std::forward<Ts>(args)...);
    default:
        THROW_HR(E_INVALIDARG);
    }
}


namespace StringifyHelpers
{
template <typename T>
inline gsl::czstring ToString(T value)
{
#ifndef WAI_BUILD_LINUX
    // Clang will instantiate this template even if it isn't used,
    // so this static_assert will always fire and break the build.
    static_assert(false, "Not implemented for this type");
#endif
}

template <>
inline gsl::czstring ToString(DML_TENSOR_DATA_TYPE value)
{
    switch (value)
    {
    case DML_TENSOR_DATA_TYPE_UNKNOWN: return "DML_TENSOR_DATA_TYPE_UNKNOWN";
    case DML_TENSOR_DATA_TYPE_FLOAT32: return "DML_TENSOR_DATA_TYPE_FLOAT32";
    case DML_TENSOR_DATA_TYPE_FLOAT16: return "DML_TENSOR_DATA_TYPE_FLOAT16";
    case DML_TENSOR_DATA_TYPE_UINT32: return "DML_TENSOR_DATA_TYPE_UINT32";
    case DML_TENSOR_DATA_TYPE_UINT16: return "DML_TENSOR_DATA_TYPE_UINT16";
    case DML_TENSOR_DATA_TYPE_UINT8: return "DML_TENSOR_DATA_TYPE_UINT8";
    case DML_TENSOR_DATA_TYPE_INT32: return "DML_TENSOR_DATA_TYPE_INT32";
    case DML_TENSOR_DATA_TYPE_INT16: return "DML_TENSOR_DATA_TYPE_INT16";
    case DML_TENSOR_DATA_TYPE_INT8: return "DML_TENSOR_DATA_TYPE_INT8";
    case DML_TENSOR_DATA_TYPE_FLOAT64: return "DML_TENSOR_DATA_TYPE_FLOAT64";
    case DML_TENSOR_DATA_TYPE_UINT64: return "DML_TENSOR_DATA_TYPE_UINT64";
    case DML_TENSOR_DATA_TYPE_INT64: return "DML_TENSOR_DATA_TYPE_INT64";
    case DML_TENSOR_DATA_TYPE_UINT4: return "DML_TENSOR_DATA_TYPE_UINT4";
    case DML_TENSOR_DATA_TYPE_INT4: return "DML_TENSOR_DATA_TYPE_INT4";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_TENSOR_TYPE value)
{
    switch (value)
    {
    case DML_TENSOR_TYPE_INVALID: return "DML_TENSOR_TYPE_INVALID";
    case DML_TENSOR_TYPE_BUFFER: return "DML_TENSOR_TYPE_BUFFER";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_OPERATOR_TYPE value)
{
    switch (value)
    {
    case DML_OPERATOR_INVALID: return "DML_OPERATOR_INVALID";
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY: return "DML_OPERATOR_ELEMENT_WISE_IDENTITY";
    case DML_OPERATOR_ELEMENT_WISE_ABS: return "DML_OPERATOR_ELEMENT_WISE_ABS";
    case DML_OPERATOR_ELEMENT_WISE_ACOS: return "DML_OPERATOR_ELEMENT_WISE_ACOS";
    case DML_OPERATOR_ELEMENT_WISE_ADD: return "DML_OPERATOR_ELEMENT_WISE_ADD";
    case DML_OPERATOR_ELEMENT_WISE_ASIN: return "DML_OPERATOR_ELEMENT_WISE_ASIN";
    case DML_OPERATOR_ELEMENT_WISE_ATAN: return "DML_OPERATOR_ELEMENT_WISE_ATAN";
    case DML_OPERATOR_ELEMENT_WISE_CEIL: return "DML_OPERATOR_ELEMENT_WISE_CEIL";
    case DML_OPERATOR_ELEMENT_WISE_CLIP: return "DML_OPERATOR_ELEMENT_WISE_CLIP";
    case DML_OPERATOR_ELEMENT_WISE_COS: return "DML_OPERATOR_ELEMENT_WISE_COS";
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE: return "DML_OPERATOR_ELEMENT_WISE_DIVIDE";
    case DML_OPERATOR_ELEMENT_WISE_EXP: return "DML_OPERATOR_ELEMENT_WISE_EXP";
    case DML_OPERATOR_ELEMENT_WISE_FLOOR: return "DML_OPERATOR_ELEMENT_WISE_FLOOR";
    case DML_OPERATOR_ELEMENT_WISE_LOG: return "DML_OPERATOR_ELEMENT_WISE_LOG";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR: return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR";
    case DML_OPERATOR_ELEMENT_WISE_MAX: return "DML_OPERATOR_ELEMENT_WISE_MAX";
    case DML_OPERATOR_ELEMENT_WISE_MEAN: return "DML_OPERATOR_ELEMENT_WISE_MEAN";
    case DML_OPERATOR_ELEMENT_WISE_MIN: return "DML_OPERATOR_ELEMENT_WISE_MIN";
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY: return "DML_OPERATOR_ELEMENT_WISE_MULTIPLY";
    case DML_OPERATOR_ELEMENT_WISE_POW: return "DML_OPERATOR_ELEMENT_WISE_POW";
    case DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW: return "DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW";
    case DML_OPERATOR_ELEMENT_WISE_RECIP: return "DML_OPERATOR_ELEMENT_WISE_RECIP";
    case DML_OPERATOR_ELEMENT_WISE_SIN: return "DML_OPERATOR_ELEMENT_WISE_SIN";
    case DML_OPERATOR_ELEMENT_WISE_SQRT: return "DML_OPERATOR_ELEMENT_WISE_SQRT";
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT: return "DML_OPERATOR_ELEMENT_WISE_SUBTRACT";
    case DML_OPERATOR_ELEMENT_WISE_TAN: return "DML_OPERATOR_ELEMENT_WISE_TAN";
    case DML_OPERATOR_ELEMENT_WISE_THRESHOLD: return "DML_OPERATOR_ELEMENT_WISE_THRESHOLD";
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR: return "DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR";
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR: return "DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR";
    case DML_OPERATOR_ACTIVATION_ELU: return "DML_OPERATOR_ACTIVATION_ELU";
    case DML_OPERATOR_ACTIVATION_CELU: return "DML_OPERATOR_ACTIVATION_CELU";
    case DML_OPERATOR_ACTIVATION_HARDMAX: return "DML_OPERATOR_ACTIVATION_HARDMAX";
    case DML_OPERATOR_ACTIVATION_HARDMAX1: return "DML_OPERATOR_ACTIVATION_HARDMAX1";
    case DML_OPERATOR_ACTIVATION_HARD_SIGMOID: return "DML_OPERATOR_ACTIVATION_HARD_SIGMOID";
    case DML_OPERATOR_ACTIVATION_IDENTITY: return "DML_OPERATOR_ACTIVATION_IDENTITY";
    case DML_OPERATOR_ACTIVATION_LEAKY_RELU: return "DML_OPERATOR_ACTIVATION_LEAKY_RELU";
    case DML_OPERATOR_ACTIVATION_LINEAR: return "DML_OPERATOR_ACTIVATION_LINEAR";
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX: return "DML_OPERATOR_ACTIVATION_LOG_SOFTMAX";
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1: return "DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1";
    case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU: return "DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU";
    case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS: return "DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS";
    case DML_OPERATOR_ACTIVATION_RELU: return "DML_OPERATOR_ACTIVATION_RELU";
    case DML_OPERATOR_ACTIVATION_SCALED_ELU: return "DML_OPERATOR_ACTIVATION_SCALED_ELU";
    case DML_OPERATOR_ACTIVATION_SCALED_TANH: return "DML_OPERATOR_ACTIVATION_SCALED_TANH";
    case DML_OPERATOR_ACTIVATION_SIGMOID: return "DML_OPERATOR_ACTIVATION_SIGMOID";
    case DML_OPERATOR_ACTIVATION_SOFTMAX: return "DML_OPERATOR_ACTIVATION_SOFTMAX";
    case DML_OPERATOR_ACTIVATION_SOFTMAX1: return "DML_OPERATOR_ACTIVATION_SOFTMAX1";
    case DML_OPERATOR_ACTIVATION_SOFTPLUS: return "DML_OPERATOR_ACTIVATION_SOFTPLUS";
    case DML_OPERATOR_ACTIVATION_SOFTSIGN: return "DML_OPERATOR_ACTIVATION_SOFTSIGN";
    case DML_OPERATOR_ACTIVATION_TANH: return "DML_OPERATOR_ACTIVATION_TANH";
    case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU: return "DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU";
    case DML_OPERATOR_CONVOLUTION: return "DML_OPERATOR_CONVOLUTION";
    case DML_OPERATOR_GEMM: return "DML_OPERATOR_GEMM";
    case DML_OPERATOR_REDUCE: return "DML_OPERATOR_REDUCE";
    case DML_OPERATOR_AVERAGE_POOLING: return "DML_OPERATOR_AVERAGE_POOLING";
    case DML_OPERATOR_AVERAGE_POOLING1: return "DML_OPERATOR_AVERAGE_POOLING1";
    case DML_OPERATOR_LP_POOLING: return "DML_OPERATOR_LP_POOLING";
    case DML_OPERATOR_LP_POOLING1: return "DML_OPERATOR_LP_POOLING1";
    case DML_OPERATOR_MAX_POOLING: return "DML_OPERATOR_MAX_POOLING";
    case DML_OPERATOR_ROI_POOLING: return "DML_OPERATOR_ROI_POOLING";
    case DML_OPERATOR_SLICE: return "DML_OPERATOR_SLICE";
    case DML_OPERATOR_CAST: return "DML_OPERATOR_CAST";
    case DML_OPERATOR_SPLIT: return "DML_OPERATOR_SPLIT";
    case DML_OPERATOR_JOIN: return "DML_OPERATOR_JOIN";
    case DML_OPERATOR_PADDING: return "DML_OPERATOR_PADDING";
    case DML_OPERATOR_PADDING1: return "DML_OPERATOR_PADDING1";
    case DML_OPERATOR_VALUE_SCALE_2D: return "DML_OPERATOR_VALUE_SCALE_2D";
    case DML_OPERATOR_UPSAMPLE_2D: return "DML_OPERATOR_UPSAMPLE_2D";
    case DML_OPERATOR_GATHER: return "DML_OPERATOR_GATHER";
    case DML_OPERATOR_SPACE_TO_DEPTH: return "DML_OPERATOR_SPACE_TO_DEPTH";
    case DML_OPERATOR_DEPTH_TO_SPACE: return "DML_OPERATOR_DEPTH_TO_SPACE";
    case DML_OPERATOR_TILE: return "DML_OPERATOR_TILE";
    case DML_OPERATOR_TOP_K: return "DML_OPERATOR_TOP_K";
    case DML_OPERATOR_BATCH_NORMALIZATION: return "DML_OPERATOR_BATCH_NORMALIZATION";
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING: return "DML_OPERATOR_BATCH_NORMALIZATION_TRAINING";
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION: return "DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION";
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION: return "DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION";
    case DML_OPERATOR_LP_NORMALIZATION: return "DML_OPERATOR_LP_NORMALIZATION";
    case DML_OPERATOR_RNN: return "DML_OPERATOR_RNN";
    case DML_OPERATOR_LSTM: return "DML_OPERATOR_LSTM";
    case DML_OPERATOR_GRU: return "DML_OPERATOR_GRU";
    case DML_OPERATOR_ELEMENT_WISE_SIGN: return "DML_OPERATOR_ELEMENT_WISE_SIGN";
    case DML_OPERATOR_ELEMENT_WISE_IS_NAN: return "DML_OPERATOR_ELEMENT_WISE_IS_NAN";
    case DML_OPERATOR_ELEMENT_WISE_ERF: return "DML_OPERATOR_ELEMENT_WISE_ERF";
    case DML_OPERATOR_ELEMENT_WISE_SINH: return "DML_OPERATOR_ELEMENT_WISE_SINH";
    case DML_OPERATOR_ELEMENT_WISE_COSH: return "DML_OPERATOR_ELEMENT_WISE_COSH";
    case DML_OPERATOR_ELEMENT_WISE_TANH: return "DML_OPERATOR_ELEMENT_WISE_TANH";
    case DML_OPERATOR_ELEMENT_WISE_ASINH: return "DML_OPERATOR_ELEMENT_WISE_ASINH";
    case DML_OPERATOR_ELEMENT_WISE_ACOSH: return "DML_OPERATOR_ELEMENT_WISE_ACOSH";
    case DML_OPERATOR_ELEMENT_WISE_ATANH: return "DML_OPERATOR_ELEMENT_WISE_ATANH";
    case DML_OPERATOR_ELEMENT_WISE_IF: return "DML_OPERATOR_ELEMENT_WISE_IF";
    case DML_OPERATOR_ELEMENT_WISE_ADD1: return "DML_OPERATOR_ELEMENT_WISE_ADD1";
    case DML_OPERATOR_ACTIVATION_SHRINK: return "DML_OPERATOR_ACTIVATION_SHRINK";
    case DML_OPERATOR_MAX_POOLING1: return "DML_OPERATOR_MAX_POOLING1";
    case DML_OPERATOR_MAX_UNPOOLING: return "DML_OPERATOR_MAX_UNPOOLING";
    case DML_OPERATOR_DIAGONAL_MATRIX: return "DML_OPERATOR_DIAGONAL_MATRIX";
    case DML_OPERATOR_SCATTER: return "DML_OPERATOR_SCATTER";
    case DML_OPERATOR_ONE_HOT: return "DML_OPERATOR_ONE_HOT";
    case DML_OPERATOR_RESAMPLE: return "DML_OPERATOR_RESAMPLE";
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT: return "DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT";
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT: return "DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT";
    case DML_OPERATOR_ELEMENT_WISE_ROUND: return "DML_OPERATOR_ELEMENT_WISE_ROUND";
    case DML_OPERATOR_ELEMENT_WISE_IS_INFINITY: return "DML_OPERATOR_ELEMENT_WISE_IS_INFINITY";
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE: return "DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE";
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR: return "DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR";
    case DML_OPERATOR_FILL_VALUE_SEQUENCE: return "DML_OPERATOR_FILL_VALUE_SEQUENCE";
    case DML_OPERATOR_FILL_VALUE_CONSTANT: return "DML_OPERATOR_FILL_VALUE_CONSTANT";
    case DML_OPERATOR_CUMULATIVE_SUMMATION: return "DML_OPERATOR_CUMULATIVE_SUMMATION";
    case DML_OPERATOR_REVERSE_SUBSEQUENCES: return "DML_OPERATOR_REVERSE_SUBSEQUENCES";
    case DML_OPERATOR_GATHER_ELEMENTS: return "DML_OPERATOR_GATHER_ELEMENTS";
    case DML_OPERATOR_GATHER_ND: return "DML_OPERATOR_GATHER_ND";
    case DML_OPERATOR_SCATTER_ND: return "DML_OPERATOR_SCATTER_ND";
    case DML_OPERATOR_MAX_POOLING2: return "DML_OPERATOR_MAX_POOLING2";
    case DML_OPERATOR_SLICE1: return "DML_OPERATOR_SLICE1";
    case DML_OPERATOR_TOP_K1: return "DML_OPERATOR_TOP_K1";
    case DML_OPERATOR_DEPTH_TO_SPACE1: return "DML_OPERATOR_DEPTH_TO_SPACE1";
    case DML_OPERATOR_SPACE_TO_DEPTH1: return "DML_OPERATOR_SPACE_TO_DEPTH1";
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1: return "DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1";
    case DML_OPERATOR_RESAMPLE1: return "DML_OPERATOR_RESAMPLE1";
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER: return "DML_OPERATOR_MATRIX_MULTIPLY_INTEGER";
    case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY: return "DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY";
    case DML_OPERATOR_CONVOLUTION_INTEGER: return "DML_OPERATOR_CONVOLUTION_INTEGER";
    case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION: return "DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION";
    case DML_OPERATOR_ELEMENT_WISE_BIT_AND: return "DML_OPERATOR_ELEMENT_WISE_BIT_AND";
    case DML_OPERATOR_ELEMENT_WISE_BIT_OR: return "DML_OPERATOR_ELEMENT_WISE_BIT_OR";
    case DML_OPERATOR_ELEMENT_WISE_BIT_XOR: return "DML_OPERATOR_ELEMENT_WISE_BIT_XOR";
    case DML_OPERATOR_ELEMENT_WISE_BIT_NOT: return "DML_OPERATOR_ELEMENT_WISE_BIT_NOT";
    case DML_OPERATOR_ELEMENT_WISE_BIT_COUNT: return "DML_OPERATOR_ELEMENT_WISE_BIT_COUNT";
    case DML_OPERATOR_ACTIVATION_RELU_GRAD: return "DML_OPERATOR_ACTIVATION_RELU_GRAD";
    case DML_OPERATOR_AVERAGE_POOLING_GRAD: return "DML_OPERATOR_AVERAGE_POOLING_GRAD";
    case DML_OPERATOR_MAX_POOLING_GRAD: return "DML_OPERATOR_MAX_POOLING_GRAD";
    case DML_OPERATOR_RANDOM_GENERATOR: return "DML_OPERATOR_RANDOM_GENERATOR";
    case DML_OPERATOR_NONZERO_COORDINATES: return "DML_OPERATOR_NONZERO_COORDINATES";
    case DML_OPERATOR_RESAMPLE_GRAD: return "DML_OPERATOR_RESAMPLE_GRAD";
    case DML_OPERATOR_SLICE_GRAD: return "DML_OPERATOR_SLICE_GRAD";
    case DML_OPERATOR_ADAM_OPTIMIZER: return "DML_OPERATOR_ADAM_OPTIMIZER";
    case DML_OPERATOR_ARGMIN: return "DML_OPERATOR_ARGMIN";
    case DML_OPERATOR_ARGMAX: return "DML_OPERATOR_ARGMAX";
    case DML_OPERATOR_ROI_ALIGN: return "DML_OPERATOR_ROI_ALIGN";
    case DML_OPERATOR_GATHER_ND1: return "DML_OPERATOR_GATHER_ND1";
    case DML_OPERATOR_ELEMENT_WISE_ATAN_YX: return "DML_OPERATOR_ELEMENT_WISE_ATAN_YX";
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD: return "DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD";
    case DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE: return "DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE";
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD: return "DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD";
    case DML_OPERATOR_CUMULATIVE_PRODUCT: return "DML_OPERATOR_CUMULATIVE_PRODUCT";
    case DML_OPERATOR_BATCH_NORMALIZATION_GRAD: return "DML_OPERATOR_BATCH_NORMALIZATION_GRAD";
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD: return "DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD";
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD: return "DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD";
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR: return "DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR";
    case DML_OPERATOR_ROI_ALIGN1: return "DML_OPERATOR_ROI_ALIGN1";
    case DML_OPERATOR_ELEMENT_WISE_CLIP1: return "DML_OPERATOR_ELEMENT_WISE_CLIP1";
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1: return "DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1";
    case DML_OPERATOR_ELEMENT_WISE_NEGATE: return "DML_OPERATOR_ELEMENT_WISE_NEGATE";
    case DML_OPERATOR_ACTIVATION_GELU: return "DML_OPERATOR_ACTIVATION_GELU";
    case DML_OPERATOR_ACTIVATION_SWISH: return "DML_OPERATOR_ACTIVATION_SWISH";
    case DML_OPERATOR_ACTIVATION_HARD_SWISH: return "DML_OPERATOR_ACTIVATION_HARD_SWISH";
    case DML_OPERATOR_RESAMPLE2: return "DML_OPERATOR_RESAMPLE2";
    case DML_OPERATOR_RESAMPLE_GRAD1: return "DML_OPERATOR_RESAMPLE_GRAD1";
    case DML_OPERATOR_DIAGONAL_MATRIX1: return "DML_OPERATOR_DIAGONAL_MATRIX1";
    case DML_OPERATOR_MULTIHEAD_ATTENTION: return "DML_OPERATOR_MULTIHEAD_ATTENTION";
    case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING: return "DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING";
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT: return "DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT";
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2: return "DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2";
    case DML_OPERATOR_MULTIHEAD_ATTENTION1: return "DML_OPERATOR_MULTIHEAD_ATTENTION1";
    case DML_OPERATOR_QUANTIZE: return "DML_OPERATOR_QUANTIZE";
    case DML_OPERATOR_DEQUANTIZE: return "DML_OPERATOR_DEQUANTIZE";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_BINDING_TYPE value)
{
    switch (value)
    {
    case DML_BINDING_TYPE_NONE: return "DML_BINDING_TYPE_NONE";
    case DML_BINDING_TYPE_BUFFER: return "DML_BINDING_TYPE_BUFFER";
    case DML_BINDING_TYPE_BUFFER_ARRAY: return "DML_BINDING_TYPE_BUFFER_ARRAY";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_REDUCE_FUNCTION value)
{
    switch (value)
    {
    case DML_REDUCE_FUNCTION_ARGMAX: return "DML_REDUCE_FUNCTION_ARGMAX";
    case DML_REDUCE_FUNCTION_ARGMIN: return "DML_REDUCE_FUNCTION_ARGMIN";
    case DML_REDUCE_FUNCTION_AVERAGE: return "DML_REDUCE_FUNCTION_AVERAGE";
    case DML_REDUCE_FUNCTION_L1: return "DML_REDUCE_FUNCTION_L1";
    case DML_REDUCE_FUNCTION_L2: return "DML_REDUCE_FUNCTION_L2";
    case DML_REDUCE_FUNCTION_LOG_SUM: return "DML_REDUCE_FUNCTION_LOG_SUM";
    case DML_REDUCE_FUNCTION_LOG_SUM_EXP: return "DML_REDUCE_FUNCTION_LOG_SUM_EXP";
    case DML_REDUCE_FUNCTION_MAX: return "DML_REDUCE_FUNCTION_MAX";
    case DML_REDUCE_FUNCTION_MIN: return "DML_REDUCE_FUNCTION_MIN";
    case DML_REDUCE_FUNCTION_MULTIPLY: return "DML_REDUCE_FUNCTION_MULTIPLY";
    case DML_REDUCE_FUNCTION_SUM: return "DML_REDUCE_FUNCTION_SUM";
    case DML_REDUCE_FUNCTION_SUM_SQUARE: return "DML_REDUCE_FUNCTION_SUM_SQUARE";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_MATRIX_TRANSFORM value)
{
    switch (value)
    {
    case DML_MATRIX_TRANSFORM_NONE: return "DML_MATRIX_TRANSFORM_NONE";
    case DML_MATRIX_TRANSFORM_TRANSPOSE: return "DML_MATRIX_TRANSFORM_TRANSPOSE";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_CONVOLUTION_MODE value)
{
    switch (value)
    {
    case DML_CONVOLUTION_MODE_CONVOLUTION: return "DML_CONVOLUTION_MODE_CONVOLUTION";
    case DML_CONVOLUTION_MODE_CROSS_CORRELATION: return "DML_CONVOLUTION_MODE_CROSS_CORRELATION";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_CONVOLUTION_DIRECTION value)
{
    switch (value)
    {
    case DML_CONVOLUTION_DIRECTION_FORWARD: return "DML_CONVOLUTION_DIRECTION_FORWARD";
    case DML_CONVOLUTION_DIRECTION_BACKWARD: return "DML_CONVOLUTION_DIRECTION_BACKWARD";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_PADDING_MODE value)
{
    switch (value)
    {
    case DML_PADDING_MODE_CONSTANT: return "DML_PADDING_MODE_CONSTANT";
    case DML_PADDING_MODE_EDGE: return "DML_PADDING_MODE_EDGE";
    case DML_PADDING_MODE_REFLECTION: return "DML_PADDING_MODE_REFLECTION";
    case DML_PADDING_MODE_SYMMETRIC: return "DML_PADDING_MODE_SYMMETRIC";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_INTERPOLATION_MODE value)
{
    switch (value)
    {
    case DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR: return "DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR";
    case DML_INTERPOLATION_MODE_LINEAR: return "DML_INTERPOLATION_MODE_LINEAR";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_RECURRENT_NETWORK_DIRECTION value)
{
    switch (value)
    {
    case DML_RECURRENT_NETWORK_DIRECTION_FORWARD: return "DML_RECURRENT_NETWORK_DIRECTION_FORWARD";
    case DML_RECURRENT_NETWORK_DIRECTION_BACKWARD: return "DML_RECURRENT_NETWORK_DIRECTION_BACKWARD";
    case DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL: return "DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_FEATURE value)
{
    switch (value)
    {
    case DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT: return "DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT";
    case DML_FEATURE_FEATURE_LEVELS: return "DML_FEATURE_FEATURE_LEVELS";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_FEATURE_LEVEL value)
{
    switch (value)
    {
    case DML_FEATURE_LEVEL_1_0: return "DML_FEATURE_LEVEL_1_0";
    case DML_FEATURE_LEVEL_2_0: return "DML_FEATURE_LEVEL_2_0";
    case DML_FEATURE_LEVEL_2_1: return "DML_FEATURE_LEVEL_2_1";
    case DML_FEATURE_LEVEL_3_0: return "DML_FEATURE_LEVEL_3_0";
    case DML_FEATURE_LEVEL_3_1: return "DML_FEATURE_LEVEL_3_1";
    case DML_FEATURE_LEVEL_4_0: return "DML_FEATURE_LEVEL_4_0";
    case DML_FEATURE_LEVEL_4_1: return "DML_FEATURE_LEVEL_4_1";
    case DML_FEATURE_LEVEL_5_0: return "DML_FEATURE_LEVEL_5_0";
    case DML_FEATURE_LEVEL_5_1: return "DML_FEATURE_LEVEL_5_1";
    case DML_FEATURE_LEVEL_5_2: return "DML_FEATURE_LEVEL_5_2";
    case DML_FEATURE_LEVEL_6_0: return "DML_FEATURE_LEVEL_6_0";
    case DML_FEATURE_LEVEL_6_1: return "DML_FEATURE_LEVEL_6_1";
    case DML_FEATURE_LEVEL_6_2: return "DML_FEATURE_LEVEL_6_2";
    case DML_FEATURE_LEVEL_6_3: return "DML_FEATURE_LEVEL_6_3";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_IS_INFINITY_MODE value)
{
    switch (value)
    {
    case DML_IS_INFINITY_MODE_EITHER: return "DML_IS_INFINITY_MODE_EITHER";
    case DML_IS_INFINITY_MODE_POSITIVE: return "DML_IS_INFINITY_MODE_POSITIVE";
    case DML_IS_INFINITY_MODE_NEGATIVE: return "DML_IS_INFINITY_MODE_NEGATIVE";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_DEPTH_SPACE_ORDER value)
{
    switch (value)
    {
    case DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW: return "DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW";
    case DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH: return "DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_AXIS_DIRECTION value)
{
    switch (value)
    {
    case DML_AXIS_DIRECTION_INCREASING: return "DML_AXIS_DIRECTION_INCREASING";
    case DML_AXIS_DIRECTION_DECREASING: return "DML_AXIS_DIRECTION_DECREASING";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_ROUNDING_MODE value)
{
    switch (value)
    {
    case DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN: return "DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN";
    case DML_ROUNDING_MODE_TOWARD_ZERO: return "DML_ROUNDING_MODE_TOWARD_ZERO";
    case DML_ROUNDING_MODE_TOWARD_INFINITY: return "DML_ROUNDING_MODE_TOWARD_INFINITY";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_RANDOM_GENERATOR_TYPE value)
{
    switch (value)
    {
    case DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10: return "DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_MULTIHEAD_ATTENTION_MASK_TYPE value)
{
    switch (value)
    {
    case DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE: return "DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE";
    case DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH: return "DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH";
    case DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START: return "DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START";
    case DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_QUERY_SEQUENCE_LENGTH_START_END: return "DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_QUERY_SEQUENCE_LENGTH_START_END";
    case DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN: return "DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN";
    default:
        assert(false);
        return "<unknown>";
    }
}

template <>
inline gsl::czstring ToString(DML_QUANTIZATION_TYPE value)
{
    switch (value)
    {
    case DML_QUANTIZATION_TYPE_NONE: return "DML_QUANTIZATION_TYPE_NONE";
    case DML_QUANTIZATION_TYPE_SCALE: return "DML_QUANTIZATION_TYPE_SCALE";
    case DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT: return "DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT";
    default:
        assert(false);
        return "<unknown>";
    }
}


template <typename T>
T FromString(std::string_view value);

}
}
