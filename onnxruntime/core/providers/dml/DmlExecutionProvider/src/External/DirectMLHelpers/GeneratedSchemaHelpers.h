﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace SchemaHelpers
{
AbstractOperatorDesc ConvertOperatorDesc(const DML_OPERATOR_DESC& opDesc);

inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_IDENTITY_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_IDENTITY_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_IDENTITY_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ABS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ABS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ABS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ABS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ACOS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ACOS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ACOS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ACOS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ADD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ADD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_ADD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_ADD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ASIN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ASIN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ASIN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ASIN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ATAN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ATAN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ATAN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ATAN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_CEIL_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_CEIL_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CEIL_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CEIL_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_CLIP_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Min))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<FLOAT>(desc.Max))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
        OperatorField(&DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.MinMaxDataType))),
        OperatorField(&DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.Min))),
        OperatorField(&DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.Max))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Min))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<FLOAT>(desc.Max))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.MinMaxDataType))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.Min))),
        OperatorField(&DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.Max))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_COS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_COS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_COS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_COS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_DIVIDE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_DIVIDE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_DIVIDE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_EXP_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_EXP_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_EXP_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_EXP_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_FLOOR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_FLOOR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_FLOOR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOG_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOG_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOG_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOG_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_MAX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_MAX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_MAX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_MAX_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_MEAN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_MEAN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_MEAN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_MEAN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_MIN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_MIN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_MIN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_MIN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_MULTIPLY_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_MULTIPLY_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_MULTIPLY_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_POW_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_POW_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_POW_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ExponentTensor))),
        OperatorField(&DML_ELEMENT_WISE_POW_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_POW_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
        OperatorField(&DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Exponent))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_RECIP_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_RECIP_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_RECIP_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_RECIP_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_SIN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_SIN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SIN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SIN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_SQRT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_SQRT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SQRT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SQRT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ATAN_YX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_ATAN_YX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_ATAN_YX_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_SUBTRACT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_SUBTRACT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_SUBTRACT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_TAN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_TAN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_TAN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_TAN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_THRESHOLD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_THRESHOLD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_THRESHOLD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
        OperatorField(&DML_ELEMENT_WISE_THRESHOLD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Min))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ZeroPointTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ZeroPointTensor))),
        OperatorField(&DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_CONVOLUTION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FilterTensor))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.Mode))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.Direction))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<const UINT*>(desc.OutputPadding), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<UINT>(desc.GroupCount))),
        OperatorField(&DML_CONVOLUTION_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_GEMM_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.CTensor))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.TransA))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.TransB))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
        OperatorField(&DML_GEMM_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_REDUCE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_REDUCE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<UINT>(desc.Function))),
        OperatorField(&DML_REDUCE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_REDUCE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_REDUCE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_REDUCE_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ARGMIN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ARGMIN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ARGMIN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ARGMIN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_ARGMIN_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
        OperatorField(&DML_ARGMIN_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.AxisDirection))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ARGMAX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ARGMAX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ARGMAX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ARGMAX_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_ARGMAX_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
        OperatorField(&DML_ARGMAX_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.AxisDirection))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_AVERAGE_POOLING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<UINT>(desc.IncludePadding))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_AVERAGE_POOLING1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING1_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<UINT>(desc.IncludePadding))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_LP_POOLING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<UINT>(desc.P))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_LP_POOLING1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
        OperatorField(&DML_LP_POOLING1_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<UINT>(desc.P))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MAX_POOLING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MAX_POOLING1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputIndicesTensor))),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ROI_POOLING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ROI_POOLING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ROI_POOLING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ROITensor))),
        OperatorField(&DML_ROI_POOLING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ROI_POOLING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScale))),
        OperatorField(&DML_ROI_POOLING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<DML_SIZE_2D>(desc.PooledSize))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SLICE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SLICE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SLICE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_SLICE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_SLICE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Offsets), desc.DimensionCount)),
        OperatorField(&DML_SLICE_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.Sizes), desc.DimensionCount)),
        OperatorField(&DML_SLICE_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_CAST_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_CAST_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_CAST_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SPLIT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SPLIT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SPLIT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<UINT>(desc.OutputCount))),
        OperatorField(&DML_SPLIT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensors), desc.OutputCount)),
        OperatorField(&DML_SPLIT_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_JOIN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_JOIN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<UINT>(desc.InputCount))),
        OperatorField(&DML_JOIN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensors), desc.InputCount)),
        OperatorField(&DML_JOIN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_JOIN_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_PADDING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.PaddingMode))),
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.PaddingValue))),
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_PADDING_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_PADDING1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.PaddingMode))),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.PaddingValueDataType))),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.PaddingValue))),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_PADDING1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_VALUE_SCALE_2D_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_VALUE_SCALE_2D_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_VALUE_SCALE_2D_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_VALUE_SCALE_2D_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Scale))),
        OperatorField(&DML_VALUE_SCALE_2D_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.ChannelCount))),
        OperatorField(&DML_VALUE_SCALE_2D_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const FLOAT*>(desc.Bias), desc.ChannelCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_UPSAMPLE_2D_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_UPSAMPLE_2D_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_UPSAMPLE_2D_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_UPSAMPLE_2D_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<DML_SIZE_2D>(desc.ScaleSize))),
        OperatorField(&DML_UPSAMPLE_2D_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_GATHER_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_GATHER_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_GATHER_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_GATHER_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_GATHER_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
        OperatorField(&DML_GATHER_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.IndexDimensions))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SPACE_TO_DEPTH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SPACE_TO_DEPTH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SPACE_TO_DEPTH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_SPACE_TO_DEPTH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.BlockSize))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_DEPTH_TO_SPACE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_DEPTH_TO_SPACE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_DEPTH_TO_SPACE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_DEPTH_TO_SPACE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.BlockSize))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_TILE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_TILE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_TILE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_TILE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.RepeatsCount))),
        OperatorField(&DML_TILE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Repeats), desc.RepeatsCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_TOP_K_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_TOP_K_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_TOP_K_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputValueTensor))),
        OperatorField(&DML_TOP_K_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputIndexTensor))),
        OperatorField(&DML_TOP_K_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
        OperatorField(&DML_TOP_K_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.K))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_BATCH_NORMALIZATION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.MeanTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.VarianceTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<UINT>(desc.Spatial))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
        OperatorField(&DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.MeanTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.VarianceTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputBiasGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.MeanTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.VarianceTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputBiasGradientTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.CrossChannel))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.NormalizeVariance))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.CrossChannel))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.LocalSize))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<FLOAT>(desc.Bias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<bool>(desc.CrossChannel))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.LocalSize))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
        OperatorField(&DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.Bias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_LP_NORMALIZATION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_LP_NORMALIZATION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_LP_NORMALIZATION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_LP_NORMALIZATION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
        OperatorField(&DML_LP_NORMALIZATION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
        OperatorField(&DML_LP_NORMALIZATION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.P))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RNN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.WeightTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.RecurrenceTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.HiddenInitTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.SequenceLengthsTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSequenceTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSingleTensor))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<UINT>(desc.ActivationDescCount))),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.ActivationDescs), desc.ActivationDescCount)),
        OperatorField(&DML_RNN_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<UINT>(desc.Direction))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_LSTM_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.WeightTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.RecurrenceTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.HiddenInitTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.CellMemInitTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.SequenceLengthsTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.PeepholeTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSequenceTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSingleTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputCellSingleTensor))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<UINT>(desc.ActivationDescCount))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.ActivationDescs), desc.ActivationDescCount)),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<UINT>(desc.Direction))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[14], ToOperatorFieldType(static_cast<FLOAT>(desc.ClipThreshold))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[15], ToOperatorFieldType(static_cast<UINT>(desc.UseClipThreshold))),
        OperatorField(&DML_LSTM_OPERATOR_SCHEMA.Fields[16], ToOperatorFieldType(static_cast<UINT>(desc.CoupleInputForget))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_GRU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.WeightTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.RecurrenceTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.HiddenInitTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.SequenceLengthsTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSequenceTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSingleTensor))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<UINT>(desc.ActivationDescCount))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.ActivationDescs), desc.ActivationDescCount)),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<UINT>(desc.Direction))),
        OperatorField(&DML_GRU_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<UINT>(desc.LinearBeforeReset))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_SIGN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_SIGN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SIGN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_IS_NAN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_IS_NAN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_NEGATE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_NEGATE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ERF_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ERF_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ERF_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ERF_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_SINH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_SINH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SINH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_SINH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_COSH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_COSH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_COSH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_COSH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_TANH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_TANH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_TANH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_TANH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ASINH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ASINH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ASINH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ASINH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ACOSH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ACOSH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ACOSH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ATANH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ATANH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ATANH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ATANH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_SCALE_BIAS*>(desc.ScaleBias))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_IF_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_IF_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ConditionTensor))),
        OperatorField(&DML_ELEMENT_WISE_IF_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_IF_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_IF_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ADD1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ADD1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_ADD1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_ADD1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ADD1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MAX_UNPOOLING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MAX_UNPOOLING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MAX_UNPOOLING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_MAX_UNPOOLING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_DIAGONAL_MATRIX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_DIAGONAL_MATRIX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_DIAGONAL_MATRIX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<INT>(desc.Offset))),
        OperatorField(&DML_DIAGONAL_MATRIX_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Value))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SCATTER_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SCATTER_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SCATTER_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_SCATTER_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.UpdatesTensor))),
        OperatorField(&DML_SCATTER_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_SCATTER_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ONE_HOT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ONE_HOT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_ONE_HOT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ValuesTensor))),
        OperatorField(&DML_ONE_HOT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ONE_HOT_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RESAMPLE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RESAMPLE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_RESAMPLE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_RESAMPLE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_RESAMPLE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.ScaleCount))),
        OperatorField(&DML_RESAMPLE_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const FLOAT*>(desc.Scales), desc.ScaleCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_ROUND_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_ROUND_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ROUND_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_ROUND_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.RoundingMode))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.InfinityMode))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_FILL_VALUE_CONSTANT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_FILL_VALUE_CONSTANT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_FILL_VALUE_CONSTANT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<UINT>(desc.ValueDataType))),
        OperatorField(&DML_FILL_VALUE_CONSTANT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.Value))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_FILL_VALUE_SEQUENCE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_FILL_VALUE_SEQUENCE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<UINT>(desc.ValueDataType))),
        OperatorField(&DML_FILL_VALUE_SEQUENCE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.ValueStart))),
        OperatorField(&DML_FILL_VALUE_SEQUENCE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.ValueDelta))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_CUMULATIVE_SUMMATION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
        OperatorField(&DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.AxisDirection))),
        OperatorField(&DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.HasExclusiveSum))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_CUMULATIVE_PRODUCT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
        OperatorField(&DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.AxisDirection))),
        OperatorField(&DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.HasExclusiveProduct))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_REVERSE_SUBSEQUENCES_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_REVERSE_SUBSEQUENCES_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.SequenceLengthsTensor))),
        OperatorField(&DML_REVERSE_SUBSEQUENCES_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_REVERSE_SUBSEQUENCES_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_GATHER_ELEMENTS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_GATHER_ELEMENTS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_GATHER_ELEMENTS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_GATHER_ELEMENTS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_GATHER_ELEMENTS_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_GATHER_ND_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_GATHER_ND_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_GATHER_ND_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_GATHER_ND_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_GATHER_ND_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.InputDimensionCount))),
        OperatorField(&DML_GATHER_ND_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.IndicesDimensionCount))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SCATTER_ND_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SCATTER_ND_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SCATTER_ND_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_SCATTER_ND_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.UpdatesTensor))),
        OperatorField(&DML_SCATTER_ND_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_SCATTER_ND_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.InputDimensionCount))),
        OperatorField(&DML_SCATTER_ND_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.IndicesDimensionCount))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MAX_POOLING2_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputIndicesTensor))),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING2_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SLICE1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SLICE1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SLICE1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_SLICE1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_SLICE1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.InputWindowOffsets), desc.DimensionCount)),
        OperatorField(&DML_SLICE1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.InputWindowSizes), desc.DimensionCount)),
        OperatorField(&DML_SLICE1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const INT*>(desc.InputWindowStrides), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_TOP_K1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_TOP_K1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_TOP_K1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputValueTensor))),
        OperatorField(&DML_TOP_K1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputIndexTensor))),
        OperatorField(&DML_TOP_K1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Axis))),
        OperatorField(&DML_TOP_K1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.K))),
        OperatorField(&DML_TOP_K1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.AxisDirection))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_DEPTH_TO_SPACE1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_DEPTH_TO_SPACE1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_DEPTH_TO_SPACE1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_DEPTH_TO_SPACE1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.BlockSize))),
        OperatorField(&DML_DEPTH_TO_SPACE1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Order))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SPACE_TO_DEPTH1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SPACE_TO_DEPTH1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_SPACE_TO_DEPTH1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_SPACE_TO_DEPTH1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.BlockSize))),
        OperatorField(&DML_SPACE_TO_DEPTH1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Order))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<UINT>(desc.NormalizeVariance))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RESAMPLE1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const FLOAT*>(desc.Scales), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const FLOAT*>(desc.InputPixelOffsets), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const FLOAT*>(desc.OutputPixelOffsets), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AZeroPointTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BZeroPointTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_CONVOLUTION_INTEGER_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputZeroPointTensor))),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FilterTensor))),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FilterZeroPointTensor))),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<UINT>(desc.GroupCount))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FilterTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FilterScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FilterZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[14], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA.Fields[15], ToOperatorFieldType(static_cast<UINT>(desc.GroupCount))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_AND_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_AND_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_AND_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_OR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_OR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_OR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_XOR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_XOR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_XOR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_NOT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_NOT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_RELU_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_RELU_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_ACTIVATION_RELU_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<UINT>(desc.IncludePadding))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MAX_POOLING_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RANDOM_GENERATOR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RANDOM_GENERATOR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputStateTensor))),
        OperatorField(&DML_RANDOM_GENERATOR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_RANDOM_GENERATOR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputStateTensor))),
        OperatorField(&DML_RANDOM_GENERATOR_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.Type))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_NONZERO_COORDINATES_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_NONZERO_COORDINATES_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_NONZERO_COORDINATES_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputCountTensor))),
        OperatorField(&DML_NONZERO_COORDINATES_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputCoordinatesTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RESAMPLE_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const FLOAT*>(desc.Scales), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const FLOAT*>(desc.InputPixelOffsets), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const FLOAT*>(desc.OutputPixelOffsets), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_SLICE_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_SLICE_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_SLICE_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_SLICE_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_SLICE_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.InputWindowOffsets), desc.DimensionCount)),
        OperatorField(&DML_SLICE_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const UINT*>(desc.InputWindowSizes), desc.DimensionCount)),
        OperatorField(&DML_SLICE_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const INT*>(desc.InputWindowStrides), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ADAM_OPTIMIZER_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputParametersTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputFirstMomentTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputSecondMomentTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.GradientTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.TrainingStepTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputParametersTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputFirstMomentTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputSecondMomentTensor))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.LearningRate))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta1))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta2))),
        OperatorField(&DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ROI_ALIGN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ROITensor))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BatchIndicesTensor))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.ReductionFunction))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScaleX))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScaleY))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.OutOfBoundsInputValue))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<UINT>(desc.MinimumSamplesPerOutput))),
        OperatorField(&DML_ROI_ALIGN_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<UINT>(desc.MaximumSamplesPerOutput))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ROI_ALIGN1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ROITensor))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BatchIndicesTensor))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.ReductionFunction))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScaleX))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScaleY))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.InputPixelOffset))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<FLOAT>(desc.OutputPixelOffset))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<FLOAT>(desc.OutOfBoundsInputValue))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<UINT>(desc.MinimumSamplesPerOutput))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<UINT>(desc.MaximumSamplesPerOutput))),
        OperatorField(&DML_ROI_ALIGN1_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<UINT>(desc.AlignRegionsToCorners))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_GATHER_ND1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_GATHER_ND1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_GATHER_ND1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.IndicesTensor))),
        OperatorField(&DML_GATHER_ND1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_GATHER_ND1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.InputDimensionCount))),
        OperatorField(&DML_GATHER_ND1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.IndicesDimensionCount))),
        OperatorField(&DML_GATHER_ND1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<UINT>(desc.BatchDimensionCount))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleTensor))),
        OperatorField(&DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputZeroPointTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AScaleTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AZeroPointTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BScaleTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BZeroPointTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputZeroPointTensor))),
        OperatorField(&DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ROI_ALIGN_GRAD_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ROITensor))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BatchIndicesTensor))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputROIGradientTensor))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<UINT>(desc.ReductionFunction))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScaleX))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<FLOAT>(desc.SpatialScaleY))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<FLOAT>(desc.InputPixelOffset))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<FLOAT>(desc.OutputPixelOffset))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<UINT>(desc.MinimumSamplesPerOutput))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<UINT>(desc.MaximumSamplesPerOutput))),
        OperatorField(&DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA.Fields[14], ToOperatorFieldType(static_cast<UINT>(desc.AlignRegionsToCorners))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.FusedAddTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputMeanTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputVarianceTensor))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
        OperatorField(&DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RESAMPLE2_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.RoundingDirection))),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const FLOAT*>(desc.Scales), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const FLOAT*>(desc.InputPixelOffsets), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE2_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const FLOAT*>(desc.OutputPixelOffsets), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_RESAMPLE_GRAD1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputGradientTensor))),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputGradientTensor))),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.InterpolationMode))),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<UINT>(desc.RoundingDirection))),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const FLOAT*>(desc.Scales), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const FLOAT*>(desc.InputPixelOffsets), desc.DimensionCount)),
        OperatorField(&DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const FLOAT*>(desc.OutputPixelOffsets), desc.DimensionCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_DIAGONAL_MATRIX1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.ValueDataType))),
        OperatorField(&DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<DML_SCALAR_UNION>(desc.Value))),
        OperatorField(&DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<INT>(desc.DiagonalFillBegin))),
        OperatorField(&DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<INT>(desc.DiagonalFillEnd))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MULTIHEAD_ATTENTION_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.QueryTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.KeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.StackedQueryKeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.StackedKeyValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.StackedQueryKeyValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.MaskTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.RelativePositionBiasTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.PastKeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.PastValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputPresentKeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputPresentValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[14], ToOperatorFieldType(static_cast<FLOAT>(desc.Scale))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[15], ToOperatorFieldType(static_cast<FLOAT>(desc.MaskFilterValue))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[16], ToOperatorFieldType(static_cast<UINT>(desc.HeadCount))),
        OperatorField(&DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA.Fields[17], ToOperatorFieldType(static_cast<UINT>(desc.MaskType))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputScaleTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputZeroPointTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<UINT>(desc.DimensionCount))),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const UINT*>(desc.Strides), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const UINT*>(desc.WindowSize), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const UINT*>(desc.StartPadding), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<const UINT*>(desc.EndPadding), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<const UINT*>(desc.Dilations), desc.DimensionCount)),
        OperatorField(&DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<UINT>(desc.IncludePadding))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ATensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AScaleTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.AZeroPointTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BScaleTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BZeroPointTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ScaleTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<UINT>(desc.UseMean))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<UINT>(desc.UseVariance))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<FLOAT>(desc.Epsilon))),
        OperatorField(&DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_OPERATOR_DESC*>(desc.FusedActivation))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.QueryTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.KeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.ValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.StackedQueryKeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.StackedKeyValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[5], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.StackedQueryKeyValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[6], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.BiasTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[7], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.MaskTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[8], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.RelativePositionBiasTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[9], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.PastKeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[10], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.PastValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[11], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.PastSequenceLengthsTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[12], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[13], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputPresentKeyTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[14], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputPresentValueTensor))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[15], ToOperatorFieldType(static_cast<FLOAT>(desc.Scale))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[16], ToOperatorFieldType(static_cast<FLOAT>(desc.MaskFilterValue))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[17], ToOperatorFieldType(static_cast<UINT>(desc.QueryHeadCount))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[18], ToOperatorFieldType(static_cast<UINT>(desc.KeyValueHeadCount))),
        OperatorField(&DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA.Fields[19], ToOperatorFieldType(static_cast<UINT>(desc.MaskType))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_QUANTIZE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_QUANTIZE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_QUANTIZE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<UINT>(desc.QuantizationType))),
        OperatorField(&DML_QUANTIZE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.QuantizationTensorCount))),
        OperatorField(&DML_QUANTIZE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.QuantizationTensors), desc.QuantizationTensorCount)),
        OperatorField(&DML_QUANTIZE_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_DEQUANTIZE_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_DEQUANTIZE_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_DEQUANTIZE_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<UINT>(desc.QuantizationType))),
        OperatorField(&DML_DEQUANTIZE_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.QuantizationTensorCount))),
        OperatorField(&DML_DEQUANTIZE_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.QuantizationTensors), desc.QuantizationTensorCount)),
        OperatorField(&DML_DEQUANTIZE_OPERATOR_SCHEMA.Fields[4], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_ELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_ELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_ELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_ELU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_CELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_CELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_CELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_CELU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_HARDMAX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_HARDMAX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_HARDMAX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_HARDMAX1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_HARDMAX1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_HARDMAX1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_HARDMAX1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_ACTIVATION_HARDMAX1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_HARD_SIGMOID_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_HARD_SIGMOID_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_HARD_SIGMOID_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_ACTIVATION_HARD_SIGMOID_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_IDENTITY_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_IDENTITY_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_IDENTITY_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_LEAKY_RELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_LEAKY_RELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_LEAKY_RELU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_LINEAR_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_LINEAR_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_LINEAR_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_LINEAR_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_ACTIVATION_LINEAR_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.SlopeTensor))),
        OperatorField(&DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_RELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_RELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_RELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SCALED_ELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SCALED_ELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_SCALED_ELU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_ACTIVATION_SCALED_ELU_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Gamma))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SCALED_TANH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SCALED_TANH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_SCALED_TANH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_ACTIVATION_SCALED_TANH_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SIGMOID_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SIGMOID_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SIGMOID_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SOFTMAX_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SOFTMAX_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SOFTMAX_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SOFTMAX1_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SOFTMAX1_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_SOFTMAX1_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<UINT>(desc.AxisCount))),
        OperatorField(&DML_ACTIVATION_SOFTMAX1_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<const UINT*>(desc.Axes), desc.AxisCount)),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SOFTPLUS_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SOFTPLUS_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_SOFTPLUS_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Steepness))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SOFTSIGN_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SOFTSIGN_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_TANH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_TANH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_TANH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SHRINK_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SHRINK_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SHRINK_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_SHRINK_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Bias))),
        OperatorField(&DML_ACTIVATION_SHRINK_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Threshold))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_GELU_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_GELU_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_GELU_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_SWISH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_SWISH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_SWISH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_SWISH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.SigmoidInputScale))),
    };
}
inline std::vector<OperatorField> GetFields(const DML_ACTIVATION_HARD_SWISH_OPERATOR_DESC& desc)
{
    return {
        OperatorField(&DML_ACTIVATION_HARD_SWISH_OPERATOR_SCHEMA.Fields[0], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.InputTensor))),
        OperatorField(&DML_ACTIVATION_HARD_SWISH_OPERATOR_SCHEMA.Fields[1], ToOperatorFieldType(static_cast<const DML_TENSOR_DESC*>(desc.OutputTensor))),
        OperatorField(&DML_ACTIVATION_HARD_SWISH_OPERATOR_SCHEMA.Fields[2], ToOperatorFieldType(static_cast<FLOAT>(desc.Alpha))),
        OperatorField(&DML_ACTIVATION_HARD_SWISH_OPERATOR_SCHEMA.Fields[3], ToOperatorFieldType(static_cast<FLOAT>(desc.Beta))),
    };
}
inline const DML_OPERATOR_SCHEMA& GetSchema(DML_OPERATOR_TYPE operatorType)
{
    switch (operatorType)
    {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY: return DML_ELEMENT_WISE_IDENTITY_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ABS: return DML_ELEMENT_WISE_ABS_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ACOS: return DML_ELEMENT_WISE_ACOS_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ADD: return DML_ELEMENT_WISE_ADD_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ASIN: return DML_ELEMENT_WISE_ASIN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ATAN: return DML_ELEMENT_WISE_ATAN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_CEIL: return DML_ELEMENT_WISE_CEIL_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_CLIP: return DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_CLIP1: return DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD: return DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1: return DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_COS: return DML_ELEMENT_WISE_COS_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE: return DML_ELEMENT_WISE_DIVIDE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_EXP: return DML_ELEMENT_WISE_EXP_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_FLOOR: return DML_ELEMENT_WISE_FLOOR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOG: return DML_ELEMENT_WISE_LOG_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND: return DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS: return DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN: return DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN: return DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL: return DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL: return DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT: return DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR: return DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR: return DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_MAX: return DML_ELEMENT_WISE_MAX_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_MEAN: return DML_ELEMENT_WISE_MEAN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_MIN: return DML_ELEMENT_WISE_MIN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY: return DML_ELEMENT_WISE_MULTIPLY_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_POW: return DML_ELEMENT_WISE_POW_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW: return DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_RECIP: return DML_ELEMENT_WISE_RECIP_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_SIN: return DML_ELEMENT_WISE_SIN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_SQRT: return DML_ELEMENT_WISE_SQRT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE: return DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ATAN_YX: return DML_ELEMENT_WISE_ATAN_YX_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT: return DML_ELEMENT_WISE_SUBTRACT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_TAN: return DML_ELEMENT_WISE_TAN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_THRESHOLD: return DML_ELEMENT_WISE_THRESHOLD_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR: return DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR: return DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_SCHEMA;
    case DML_OPERATOR_CONVOLUTION: return DML_CONVOLUTION_OPERATOR_SCHEMA;
    case DML_OPERATOR_GEMM: return DML_GEMM_OPERATOR_SCHEMA;
    case DML_OPERATOR_REDUCE: return DML_REDUCE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ARGMIN: return DML_ARGMIN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ARGMAX: return DML_ARGMAX_OPERATOR_SCHEMA;
    case DML_OPERATOR_AVERAGE_POOLING: return DML_AVERAGE_POOLING_OPERATOR_SCHEMA;
    case DML_OPERATOR_AVERAGE_POOLING1: return DML_AVERAGE_POOLING1_OPERATOR_SCHEMA;
    case DML_OPERATOR_LP_POOLING: return DML_LP_POOLING_OPERATOR_SCHEMA;
    case DML_OPERATOR_LP_POOLING1: return DML_LP_POOLING1_OPERATOR_SCHEMA;
    case DML_OPERATOR_MAX_POOLING: return DML_MAX_POOLING_OPERATOR_SCHEMA;
    case DML_OPERATOR_MAX_POOLING1: return DML_MAX_POOLING1_OPERATOR_SCHEMA;
    case DML_OPERATOR_ROI_POOLING: return DML_ROI_POOLING_OPERATOR_SCHEMA;
    case DML_OPERATOR_SLICE: return DML_SLICE_OPERATOR_SCHEMA;
    case DML_OPERATOR_CAST: return DML_CAST_OPERATOR_SCHEMA;
    case DML_OPERATOR_SPLIT: return DML_SPLIT_OPERATOR_SCHEMA;
    case DML_OPERATOR_JOIN: return DML_JOIN_OPERATOR_SCHEMA;
    case DML_OPERATOR_PADDING: return DML_PADDING_OPERATOR_SCHEMA;
    case DML_OPERATOR_PADDING1: return DML_PADDING1_OPERATOR_SCHEMA;
    case DML_OPERATOR_VALUE_SCALE_2D: return DML_VALUE_SCALE_2D_OPERATOR_SCHEMA;
    case DML_OPERATOR_UPSAMPLE_2D: return DML_UPSAMPLE_2D_OPERATOR_SCHEMA;
    case DML_OPERATOR_GATHER: return DML_GATHER_OPERATOR_SCHEMA;
    case DML_OPERATOR_SPACE_TO_DEPTH: return DML_SPACE_TO_DEPTH_OPERATOR_SCHEMA;
    case DML_OPERATOR_DEPTH_TO_SPACE: return DML_DEPTH_TO_SPACE_OPERATOR_SCHEMA;
    case DML_OPERATOR_TILE: return DML_TILE_OPERATOR_SCHEMA;
    case DML_OPERATOR_TOP_K: return DML_TOP_K_OPERATOR_SCHEMA;
    case DML_OPERATOR_BATCH_NORMALIZATION: return DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA;
    case DML_OPERATOR_BATCH_NORMALIZATION_GRAD: return DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD: return DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION: return DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA;
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION: return DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA;
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD: return DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_LP_NORMALIZATION: return DML_LP_NORMALIZATION_OPERATOR_SCHEMA;
    case DML_OPERATOR_RNN: return DML_RNN_OPERATOR_SCHEMA;
    case DML_OPERATOR_LSTM: return DML_LSTM_OPERATOR_SCHEMA;
    case DML_OPERATOR_GRU: return DML_GRU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_SIGN: return DML_ELEMENT_WISE_SIGN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_IS_NAN: return DML_ELEMENT_WISE_IS_NAN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_NEGATE: return DML_ELEMENT_WISE_NEGATE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ERF: return DML_ELEMENT_WISE_ERF_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_SINH: return DML_ELEMENT_WISE_SINH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_COSH: return DML_ELEMENT_WISE_COSH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_TANH: return DML_ELEMENT_WISE_TANH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ASINH: return DML_ELEMENT_WISE_ASINH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ACOSH: return DML_ELEMENT_WISE_ACOSH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ATANH: return DML_ELEMENT_WISE_ATANH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_IF: return DML_ELEMENT_WISE_IF_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ADD1: return DML_ELEMENT_WISE_ADD1_OPERATOR_SCHEMA;
    case DML_OPERATOR_MAX_UNPOOLING: return DML_MAX_UNPOOLING_OPERATOR_SCHEMA;
    case DML_OPERATOR_DIAGONAL_MATRIX: return DML_DIAGONAL_MATRIX_OPERATOR_SCHEMA;
    case DML_OPERATOR_SCATTER: return DML_SCATTER_OPERATOR_SCHEMA;
    case DML_OPERATOR_ONE_HOT: return DML_ONE_HOT_OPERATOR_SCHEMA;
    case DML_OPERATOR_RESAMPLE: return DML_RESAMPLE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT: return DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT: return DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_ROUND: return DML_ELEMENT_WISE_ROUND_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_IS_INFINITY: return DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE: return DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR: return DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_SCHEMA;
    case DML_OPERATOR_FILL_VALUE_CONSTANT: return DML_FILL_VALUE_CONSTANT_OPERATOR_SCHEMA;
    case DML_OPERATOR_FILL_VALUE_SEQUENCE: return DML_FILL_VALUE_SEQUENCE_OPERATOR_SCHEMA;
    case DML_OPERATOR_CUMULATIVE_SUMMATION: return DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA;
    case DML_OPERATOR_CUMULATIVE_PRODUCT: return DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA;
    case DML_OPERATOR_REVERSE_SUBSEQUENCES: return DML_REVERSE_SUBSEQUENCES_OPERATOR_SCHEMA;
    case DML_OPERATOR_GATHER_ELEMENTS: return DML_GATHER_ELEMENTS_OPERATOR_SCHEMA;
    case DML_OPERATOR_GATHER_ND: return DML_GATHER_ND_OPERATOR_SCHEMA;
    case DML_OPERATOR_SCATTER_ND: return DML_SCATTER_ND_OPERATOR_SCHEMA;
    case DML_OPERATOR_MAX_POOLING2: return DML_MAX_POOLING2_OPERATOR_SCHEMA;
    case DML_OPERATOR_SLICE1: return DML_SLICE1_OPERATOR_SCHEMA;
    case DML_OPERATOR_TOP_K1: return DML_TOP_K1_OPERATOR_SCHEMA;
    case DML_OPERATOR_DEPTH_TO_SPACE1: return DML_DEPTH_TO_SPACE1_OPERATOR_SCHEMA;
    case DML_OPERATOR_SPACE_TO_DEPTH1: return DML_SPACE_TO_DEPTH1_OPERATOR_SCHEMA;
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1: return DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA;
    case DML_OPERATOR_RESAMPLE1: return DML_RESAMPLE1_OPERATOR_SCHEMA;
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER: return DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA;
    case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY: return DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA;
    case DML_OPERATOR_CONVOLUTION_INTEGER: return DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA;
    case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION: return DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_AND: return DML_ELEMENT_WISE_BIT_AND_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_OR: return DML_ELEMENT_WISE_BIT_OR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_XOR: return DML_ELEMENT_WISE_BIT_XOR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_NOT: return DML_ELEMENT_WISE_BIT_NOT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_BIT_COUNT: return DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_RELU_GRAD: return DML_ACTIVATION_RELU_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_AVERAGE_POOLING_GRAD: return DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_MAX_POOLING_GRAD: return DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_RANDOM_GENERATOR: return DML_RANDOM_GENERATOR_OPERATOR_SCHEMA;
    case DML_OPERATOR_NONZERO_COORDINATES: return DML_NONZERO_COORDINATES_OPERATOR_SCHEMA;
    case DML_OPERATOR_RESAMPLE_GRAD: return DML_RESAMPLE_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_SLICE_GRAD: return DML_SLICE_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_ADAM_OPTIMIZER: return DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA;
    case DML_OPERATOR_ROI_ALIGN: return DML_ROI_ALIGN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ROI_ALIGN1: return DML_ROI_ALIGN1_OPERATOR_SCHEMA;
    case DML_OPERATOR_GATHER_ND1: return DML_GATHER_ND1_OPERATOR_SCHEMA;
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR: return DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD: return DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA;
    case DML_OPERATOR_ROI_ALIGN_GRAD: return DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA;
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING: return DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA;
    case DML_OPERATOR_RESAMPLE2: return DML_RESAMPLE2_OPERATOR_SCHEMA;
    case DML_OPERATOR_RESAMPLE_GRAD1: return DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA;
    case DML_OPERATOR_DIAGONAL_MATRIX1: return DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA;
    case DML_OPERATOR_MULTIHEAD_ATTENTION: return DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA;
    case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING: return DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA;
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT: return DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA;
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2: return DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA;
    case DML_OPERATOR_MULTIHEAD_ATTENTION1: return DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA;
    case DML_OPERATOR_QUANTIZE: return DML_QUANTIZE_OPERATOR_SCHEMA;
    case DML_OPERATOR_DEQUANTIZE: return DML_DEQUANTIZE_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_ELU: return DML_ACTIVATION_ELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_CELU: return DML_ACTIVATION_CELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_HARDMAX: return DML_ACTIVATION_HARDMAX_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_HARDMAX1: return DML_ACTIVATION_HARDMAX1_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_HARD_SIGMOID: return DML_ACTIVATION_HARD_SIGMOID_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_IDENTITY: return DML_ACTIVATION_IDENTITY_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_LEAKY_RELU: return DML_ACTIVATION_LEAKY_RELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_LINEAR: return DML_ACTIVATION_LINEAR_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX: return DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1: return DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU: return DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS: return DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_RELU: return DML_ACTIVATION_RELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SCALED_ELU: return DML_ACTIVATION_SCALED_ELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SCALED_TANH: return DML_ACTIVATION_SCALED_TANH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SIGMOID: return DML_ACTIVATION_SIGMOID_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SOFTMAX: return DML_ACTIVATION_SOFTMAX_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SOFTMAX1: return DML_ACTIVATION_SOFTMAX1_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SOFTPLUS: return DML_ACTIVATION_SOFTPLUS_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SOFTSIGN: return DML_ACTIVATION_SOFTSIGN_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_TANH: return DML_ACTIVATION_TANH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU: return DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SHRINK: return DML_ACTIVATION_SHRINK_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_GELU: return DML_ACTIVATION_GELU_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_SWISH: return DML_ACTIVATION_SWISH_OPERATOR_SCHEMA;
    case DML_OPERATOR_ACTIVATION_HARD_SWISH: return DML_ACTIVATION_HARD_SWISH_OPERATOR_SCHEMA;

    default: THROW_HR(E_INVALIDARG);
    }
}

inline const bool IsValidOperator(DML_OPERATOR_TYPE operatorType)
{
    switch (operatorType)
    {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
    case DML_OPERATOR_ELEMENT_WISE_ABS:
    case DML_OPERATOR_ELEMENT_WISE_ACOS:
    case DML_OPERATOR_ELEMENT_WISE_ADD:
    case DML_OPERATOR_ELEMENT_WISE_ASIN:
    case DML_OPERATOR_ELEMENT_WISE_ATAN:
    case DML_OPERATOR_ELEMENT_WISE_CEIL:
    case DML_OPERATOR_ELEMENT_WISE_CLIP:
    case DML_OPERATOR_ELEMENT_WISE_CLIP1:
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD:
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1:
    case DML_OPERATOR_ELEMENT_WISE_COS:
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE:
    case DML_OPERATOR_ELEMENT_WISE_EXP:
    case DML_OPERATOR_ELEMENT_WISE_FLOOR:
    case DML_OPERATOR_ELEMENT_WISE_LOG:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR:
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR:
    case DML_OPERATOR_ELEMENT_WISE_MAX:
    case DML_OPERATOR_ELEMENT_WISE_MEAN:
    case DML_OPERATOR_ELEMENT_WISE_MIN:
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY:
    case DML_OPERATOR_ELEMENT_WISE_POW:
    case DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW:
    case DML_OPERATOR_ELEMENT_WISE_RECIP:
    case DML_OPERATOR_ELEMENT_WISE_SIN:
    case DML_OPERATOR_ELEMENT_WISE_SQRT:
    case DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE:
    case DML_OPERATOR_ELEMENT_WISE_ATAN_YX:
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT:
    case DML_OPERATOR_ELEMENT_WISE_TAN:
    case DML_OPERATOR_ELEMENT_WISE_THRESHOLD:
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR:
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR:
    case DML_OPERATOR_CONVOLUTION:
    case DML_OPERATOR_GEMM:
    case DML_OPERATOR_REDUCE:
    case DML_OPERATOR_ARGMIN:
    case DML_OPERATOR_ARGMAX:
    case DML_OPERATOR_AVERAGE_POOLING:
    case DML_OPERATOR_AVERAGE_POOLING1:
    case DML_OPERATOR_LP_POOLING:
    case DML_OPERATOR_LP_POOLING1:
    case DML_OPERATOR_MAX_POOLING:
    case DML_OPERATOR_MAX_POOLING1:
    case DML_OPERATOR_ROI_POOLING:
    case DML_OPERATOR_SLICE:
    case DML_OPERATOR_CAST:
    case DML_OPERATOR_SPLIT:
    case DML_OPERATOR_JOIN:
    case DML_OPERATOR_PADDING:
    case DML_OPERATOR_PADDING1:
    case DML_OPERATOR_VALUE_SCALE_2D:
    case DML_OPERATOR_UPSAMPLE_2D:
    case DML_OPERATOR_GATHER:
    case DML_OPERATOR_SPACE_TO_DEPTH:
    case DML_OPERATOR_DEPTH_TO_SPACE:
    case DML_OPERATOR_TILE:
    case DML_OPERATOR_TOP_K:
    case DML_OPERATOR_BATCH_NORMALIZATION:
    case DML_OPERATOR_BATCH_NORMALIZATION_GRAD:
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD:
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION:
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION:
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD:
    case DML_OPERATOR_LP_NORMALIZATION:
    case DML_OPERATOR_RNN:
    case DML_OPERATOR_LSTM:
    case DML_OPERATOR_GRU:
    case DML_OPERATOR_ELEMENT_WISE_SIGN:
    case DML_OPERATOR_ELEMENT_WISE_IS_NAN:
    case DML_OPERATOR_ELEMENT_WISE_NEGATE:
    case DML_OPERATOR_ELEMENT_WISE_ERF:
    case DML_OPERATOR_ELEMENT_WISE_SINH:
    case DML_OPERATOR_ELEMENT_WISE_COSH:
    case DML_OPERATOR_ELEMENT_WISE_TANH:
    case DML_OPERATOR_ELEMENT_WISE_ASINH:
    case DML_OPERATOR_ELEMENT_WISE_ACOSH:
    case DML_OPERATOR_ELEMENT_WISE_ATANH:
    case DML_OPERATOR_ELEMENT_WISE_IF:
    case DML_OPERATOR_ELEMENT_WISE_ADD1:
    case DML_OPERATOR_MAX_UNPOOLING:
    case DML_OPERATOR_DIAGONAL_MATRIX:
    case DML_OPERATOR_SCATTER:
    case DML_OPERATOR_ONE_HOT:
    case DML_OPERATOR_RESAMPLE:
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT:
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT:
    case DML_OPERATOR_ELEMENT_WISE_ROUND:
    case DML_OPERATOR_ELEMENT_WISE_IS_INFINITY:
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE:
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR:
    case DML_OPERATOR_FILL_VALUE_CONSTANT:
    case DML_OPERATOR_FILL_VALUE_SEQUENCE:
    case DML_OPERATOR_CUMULATIVE_SUMMATION:
    case DML_OPERATOR_CUMULATIVE_PRODUCT:
    case DML_OPERATOR_REVERSE_SUBSEQUENCES:
    case DML_OPERATOR_GATHER_ELEMENTS:
    case DML_OPERATOR_GATHER_ND:
    case DML_OPERATOR_SCATTER_ND:
    case DML_OPERATOR_MAX_POOLING2:
    case DML_OPERATOR_SLICE1:
    case DML_OPERATOR_TOP_K1:
    case DML_OPERATOR_DEPTH_TO_SPACE1:
    case DML_OPERATOR_SPACE_TO_DEPTH1:
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1:
    case DML_OPERATOR_RESAMPLE1:
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER:
    case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY:
    case DML_OPERATOR_CONVOLUTION_INTEGER:
    case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION:
    case DML_OPERATOR_ELEMENT_WISE_BIT_AND:
    case DML_OPERATOR_ELEMENT_WISE_BIT_OR:
    case DML_OPERATOR_ELEMENT_WISE_BIT_XOR:
    case DML_OPERATOR_ELEMENT_WISE_BIT_NOT:
    case DML_OPERATOR_ELEMENT_WISE_BIT_COUNT:
    case DML_OPERATOR_ACTIVATION_RELU_GRAD:
    case DML_OPERATOR_AVERAGE_POOLING_GRAD:
    case DML_OPERATOR_MAX_POOLING_GRAD:
    case DML_OPERATOR_RANDOM_GENERATOR:
    case DML_OPERATOR_NONZERO_COORDINATES:
    case DML_OPERATOR_RESAMPLE_GRAD:
    case DML_OPERATOR_SLICE_GRAD:
    case DML_OPERATOR_ADAM_OPTIMIZER:
    case DML_OPERATOR_ROI_ALIGN:
    case DML_OPERATOR_ROI_ALIGN1:
    case DML_OPERATOR_GATHER_ND1:
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR:
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD:
    case DML_OPERATOR_ROI_ALIGN_GRAD:
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING:
    case DML_OPERATOR_RESAMPLE2:
    case DML_OPERATOR_RESAMPLE_GRAD1:
    case DML_OPERATOR_DIAGONAL_MATRIX1:
    case DML_OPERATOR_MULTIHEAD_ATTENTION:
    case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING:
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT:
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2:
    case DML_OPERATOR_MULTIHEAD_ATTENTION1:
    case DML_OPERATOR_QUANTIZE:
    case DML_OPERATOR_DEQUANTIZE:
    case DML_OPERATOR_ACTIVATION_ELU:
    case DML_OPERATOR_ACTIVATION_CELU:
    case DML_OPERATOR_ACTIVATION_HARDMAX:
    case DML_OPERATOR_ACTIVATION_HARDMAX1:
    case DML_OPERATOR_ACTIVATION_HARD_SIGMOID:
    case DML_OPERATOR_ACTIVATION_IDENTITY:
    case DML_OPERATOR_ACTIVATION_LEAKY_RELU:
    case DML_OPERATOR_ACTIVATION_LINEAR:
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1:
    case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU:
    case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS:
    case DML_OPERATOR_ACTIVATION_RELU:
    case DML_OPERATOR_ACTIVATION_SCALED_ELU:
    case DML_OPERATOR_ACTIVATION_SCALED_TANH:
    case DML_OPERATOR_ACTIVATION_SIGMOID:
    case DML_OPERATOR_ACTIVATION_SOFTMAX:
    case DML_OPERATOR_ACTIVATION_SOFTMAX1:
    case DML_OPERATOR_ACTIVATION_SOFTPLUS:
    case DML_OPERATOR_ACTIVATION_SOFTSIGN:
    case DML_OPERATOR_ACTIVATION_TANH:
    case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU:
    case DML_OPERATOR_ACTIVATION_SHRINK:
    case DML_OPERATOR_ACTIVATION_GELU:
    case DML_OPERATOR_ACTIVATION_SWISH:
    case DML_OPERATOR_ACTIVATION_HARD_SWISH:
        return true;
    default: return false;
    }
}

inline AbstractOperatorDesc ConvertOperatorDesc(const DML_OPERATOR_DESC& opDesc)
{
    switch (static_cast<uint32_t>(opDesc.Type))
    {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_IDENTITY_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ABS:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ABS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ABS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ACOS:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ACOS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ACOS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ADD:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ADD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ADD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ASIN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ASIN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ASIN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ATAN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ATAN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ATAN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_CEIL:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_CEIL_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_CEIL_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_CLIP:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_CLIP_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_CLIP_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_CLIP1:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_CLIP1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_CLIP_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_CLIP_GRAD1:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_CLIP_GRAD1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_COS:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_COS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_COS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_DIVIDE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_EXP:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_EXP_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_EXP_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_FLOOR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_FLOOR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOG:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOG_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOG_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_MAX:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_MAX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_MAX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_MEAN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_MEAN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_MEAN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_MIN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_MIN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_MIN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_MULTIPLY_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_POW:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_POW_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_POW_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_RECIP:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_RECIP_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_RECIP_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_SIN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_SIN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_SIN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_SQRT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_SQRT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_SQRT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_DIFFERENCE_SQUARE:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_DIFFERENCE_SQUARE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ATAN_YX:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ATAN_YX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ATAN_YX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_SUBTRACT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_TAN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_TAN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_TAN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_THRESHOLD:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_THRESHOLD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_THRESHOLD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_CONVOLUTION:
        return AbstractOperatorDesc(
            &DML_CONVOLUTION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_CONVOLUTION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_GEMM:
        return AbstractOperatorDesc(
            &DML_GEMM_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_GEMM_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_REDUCE:
        return AbstractOperatorDesc(
            &DML_REDUCE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_REDUCE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ARGMIN:
        return AbstractOperatorDesc(
            &DML_ARGMIN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ARGMIN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ARGMAX:
        return AbstractOperatorDesc(
            &DML_ARGMAX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ARGMAX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_AVERAGE_POOLING:
        return AbstractOperatorDesc(
            &DML_AVERAGE_POOLING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_AVERAGE_POOLING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_AVERAGE_POOLING1:
        return AbstractOperatorDesc(
            &DML_AVERAGE_POOLING1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_AVERAGE_POOLING1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_LP_POOLING:
        return AbstractOperatorDesc(
            &DML_LP_POOLING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_LP_POOLING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_LP_POOLING1:
        return AbstractOperatorDesc(
            &DML_LP_POOLING1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_LP_POOLING1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MAX_POOLING:
        return AbstractOperatorDesc(
            &DML_MAX_POOLING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MAX_POOLING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MAX_POOLING1:
        return AbstractOperatorDesc(
            &DML_MAX_POOLING1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MAX_POOLING1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ROI_POOLING:
        return AbstractOperatorDesc(
            &DML_ROI_POOLING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ROI_POOLING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SLICE:
        return AbstractOperatorDesc(
            &DML_SLICE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SLICE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_CAST:
        return AbstractOperatorDesc(
            &DML_CAST_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_CAST_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SPLIT:
        return AbstractOperatorDesc(
            &DML_SPLIT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SPLIT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_JOIN:
        return AbstractOperatorDesc(
            &DML_JOIN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_JOIN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_PADDING:
        return AbstractOperatorDesc(
            &DML_PADDING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_PADDING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_PADDING1:
        return AbstractOperatorDesc(
            &DML_PADDING1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_PADDING1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_VALUE_SCALE_2D:
        return AbstractOperatorDesc(
            &DML_VALUE_SCALE_2D_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_VALUE_SCALE_2D_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_UPSAMPLE_2D:
        return AbstractOperatorDesc(
            &DML_UPSAMPLE_2D_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_UPSAMPLE_2D_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_GATHER:
        return AbstractOperatorDesc(
            &DML_GATHER_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_GATHER_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SPACE_TO_DEPTH:
        return AbstractOperatorDesc(
            &DML_SPACE_TO_DEPTH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SPACE_TO_DEPTH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_DEPTH_TO_SPACE:
        return AbstractOperatorDesc(
            &DML_DEPTH_TO_SPACE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_DEPTH_TO_SPACE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_TILE:
        return AbstractOperatorDesc(
            &DML_TILE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_TILE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_TOP_K:
        return AbstractOperatorDesc(
            &DML_TOP_K_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_TOP_K_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_BATCH_NORMALIZATION:
        return AbstractOperatorDesc(
            &DML_BATCH_NORMALIZATION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_BATCH_NORMALIZATION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_BATCH_NORMALIZATION_GRAD:
        return AbstractOperatorDesc(
            &DML_BATCH_NORMALIZATION_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_BATCH_NORMALIZATION_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING_GRAD:
        return AbstractOperatorDesc(
            &DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_BATCH_NORMALIZATION_TRAINING_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION:
        return AbstractOperatorDesc(
            &DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION:
        return AbstractOperatorDesc(
            &DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD:
        return AbstractOperatorDesc(
            &DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_LP_NORMALIZATION:
        return AbstractOperatorDesc(
            &DML_LP_NORMALIZATION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_LP_NORMALIZATION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RNN:
        return AbstractOperatorDesc(
            &DML_RNN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RNN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_LSTM:
        return AbstractOperatorDesc(
            &DML_LSTM_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_LSTM_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_GRU:
        return AbstractOperatorDesc(
            &DML_GRU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_GRU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_SIGN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_SIGN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_SIGN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_IS_NAN:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_IS_NAN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_NEGATE:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_NEGATE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ERF:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ERF_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ERF_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_SINH:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_SINH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_SINH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_COSH:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_COSH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_COSH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_TANH:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_TANH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_TANH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ASINH:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ASINH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ASINH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ACOSH:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ACOSH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ATANH:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ATANH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ATANH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_IF:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_IF_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_IF_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ADD1:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ADD1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ADD1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MAX_UNPOOLING:
        return AbstractOperatorDesc(
            &DML_MAX_UNPOOLING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MAX_UNPOOLING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_DIAGONAL_MATRIX:
        return AbstractOperatorDesc(
            &DML_DIAGONAL_MATRIX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_DIAGONAL_MATRIX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SCATTER:
        return AbstractOperatorDesc(
            &DML_SCATTER_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SCATTER_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ONE_HOT:
        return AbstractOperatorDesc(
            &DML_ONE_HOT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ONE_HOT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RESAMPLE:
        return AbstractOperatorDesc(
            &DML_RESAMPLE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RESAMPLE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_ROUND:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_ROUND_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_ROUND_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_IS_INFINITY:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_FILL_VALUE_CONSTANT:
        return AbstractOperatorDesc(
            &DML_FILL_VALUE_CONSTANT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_FILL_VALUE_CONSTANT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_FILL_VALUE_SEQUENCE:
        return AbstractOperatorDesc(
            &DML_FILL_VALUE_SEQUENCE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_CUMULATIVE_SUMMATION:
        return AbstractOperatorDesc(
            &DML_CUMULATIVE_SUMMATION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_CUMULATIVE_SUMMATION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_CUMULATIVE_PRODUCT:
        return AbstractOperatorDesc(
            &DML_CUMULATIVE_PRODUCT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_CUMULATIVE_PRODUCT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_REVERSE_SUBSEQUENCES:
        return AbstractOperatorDesc(
            &DML_REVERSE_SUBSEQUENCES_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_GATHER_ELEMENTS:
        return AbstractOperatorDesc(
            &DML_GATHER_ELEMENTS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_GATHER_ELEMENTS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_GATHER_ND:
        return AbstractOperatorDesc(
            &DML_GATHER_ND_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_GATHER_ND_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SCATTER_ND:
        return AbstractOperatorDesc(
            &DML_SCATTER_ND_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SCATTER_ND_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MAX_POOLING2:
        return AbstractOperatorDesc(
            &DML_MAX_POOLING2_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MAX_POOLING2_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SLICE1:
        return AbstractOperatorDesc(
            &DML_SLICE1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SLICE1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_TOP_K1:
        return AbstractOperatorDesc(
            &DML_TOP_K1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_TOP_K1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_DEPTH_TO_SPACE1:
        return AbstractOperatorDesc(
            &DML_DEPTH_TO_SPACE1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_DEPTH_TO_SPACE1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SPACE_TO_DEPTH1:
        return AbstractOperatorDesc(
            &DML_SPACE_TO_DEPTH1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SPACE_TO_DEPTH1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1:
        return AbstractOperatorDesc(
            &DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RESAMPLE1:
        return AbstractOperatorDesc(
            &DML_RESAMPLE1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RESAMPLE1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER:
        return AbstractOperatorDesc(
            &DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY:
        return AbstractOperatorDesc(
            &DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_CONVOLUTION_INTEGER:
        return AbstractOperatorDesc(
            &DML_CONVOLUTION_INTEGER_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_CONVOLUTION_INTEGER_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION:
        return AbstractOperatorDesc(
            &DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_AND:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_AND_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_OR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_OR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_XOR:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_XOR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_NOT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_NOT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_BIT_COUNT:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_RELU_GRAD:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_RELU_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_AVERAGE_POOLING_GRAD:
        return AbstractOperatorDesc(
            &DML_AVERAGE_POOLING_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MAX_POOLING_GRAD:
        return AbstractOperatorDesc(
            &DML_MAX_POOLING_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MAX_POOLING_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RANDOM_GENERATOR:
        return AbstractOperatorDesc(
            &DML_RANDOM_GENERATOR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RANDOM_GENERATOR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_NONZERO_COORDINATES:
        return AbstractOperatorDesc(
            &DML_NONZERO_COORDINATES_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_NONZERO_COORDINATES_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RESAMPLE_GRAD:
        return AbstractOperatorDesc(
            &DML_RESAMPLE_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RESAMPLE_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_SLICE_GRAD:
        return AbstractOperatorDesc(
            &DML_SLICE_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_SLICE_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ADAM_OPTIMIZER:
        return AbstractOperatorDesc(
            &DML_ADAM_OPTIMIZER_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ADAM_OPTIMIZER_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ROI_ALIGN:
        return AbstractOperatorDesc(
            &DML_ROI_ALIGN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ROI_ALIGN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ROI_ALIGN1:
        return AbstractOperatorDesc(
            &DML_ROI_ALIGN1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ROI_ALIGN1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_GATHER_ND1:
        return AbstractOperatorDesc(
            &DML_GATHER_ND1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_GATHER_ND1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR:
        return AbstractOperatorDesc(
            &DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD:
        return AbstractOperatorDesc(
            &DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ROI_ALIGN_GRAD:
        return AbstractOperatorDesc(
            &DML_ROI_ALIGN_GRAD_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ROI_ALIGN_GRAD_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_BATCH_NORMALIZATION_TRAINING:
        return AbstractOperatorDesc(
            &DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_BATCH_NORMALIZATION_TRAINING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RESAMPLE2:
        return AbstractOperatorDesc(
            &DML_RESAMPLE2_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RESAMPLE2_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_RESAMPLE_GRAD1:
        return AbstractOperatorDesc(
            &DML_RESAMPLE_GRAD1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_RESAMPLE_GRAD1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_DIAGONAL_MATRIX1:
        return AbstractOperatorDesc(
            &DML_DIAGONAL_MATRIX1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_DIAGONAL_MATRIX1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MULTIHEAD_ATTENTION:
        return AbstractOperatorDesc(
            &DML_MULTIHEAD_ATTENTION_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MULTIHEAD_ATTENTION_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING:
        return AbstractOperatorDesc(
            &DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT:
        return AbstractOperatorDesc(
            &DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2:
        return AbstractOperatorDesc(
            &DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_MULTIHEAD_ATTENTION1:
        return AbstractOperatorDesc(
            &DML_MULTIHEAD_ATTENTION1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_QUANTIZE:
        return AbstractOperatorDesc(
            &DML_QUANTIZE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_QUANTIZE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_DEQUANTIZE:
        return AbstractOperatorDesc(
            &DML_DEQUANTIZE_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_DEQUANTIZE_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_ELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_ELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_ELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_CELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_CELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_CELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_HARDMAX:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_HARDMAX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_HARDMAX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_HARDMAX1:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_HARDMAX1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_HARDMAX1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_HARD_SIGMOID:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_HARD_SIGMOID_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_IDENTITY:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_IDENTITY_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_IDENTITY_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_LEAKY_RELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_LEAKY_RELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_LINEAR:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_LINEAR_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_LINEAR_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_RELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_RELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_RELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SCALED_ELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SCALED_ELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SCALED_TANH:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SCALED_TANH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SIGMOID:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SIGMOID_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SIGMOID_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SOFTMAX:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SOFTMAX_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SOFTMAX_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SOFTMAX1:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SOFTMAX1_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SOFTPLUS:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SOFTPLUS_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SOFTSIGN:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SOFTSIGN_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_TANH:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_TANH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_TANH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SHRINK:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SHRINK_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SHRINK_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_GELU:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_GELU_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_GELU_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_SWISH:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_SWISH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_SWISH_OPERATOR_DESC*>(opDesc.Desc)));
    case DML_OPERATOR_ACTIVATION_HARD_SWISH:
        return AbstractOperatorDesc(
            &DML_ACTIVATION_HARD_SWISH_OPERATOR_SCHEMA,
            GetFields(*static_cast<const DML_ACTIVATION_HARD_SWISH_OPERATOR_DESC*>(opDesc.Desc)));
    default: THROW_HR(E_INVALIDARG);
    }

}

template<typename T>
inline AbstractOperatorDesc ConvertTypedOperatorDesc(const T& opDesc)
{
    DML_OPERATOR_DESC desc = { (DML_OPERATOR_TYPE) ApiTraits::OperatorDescTraits<T>::Type, (void*) &opDesc };
    return ConvertOperatorDesc(desc);
}
}
