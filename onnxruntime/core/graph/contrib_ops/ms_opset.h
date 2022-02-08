// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnx/defs/schema.h"
#include "core/graph/contrib_ops/ms_schema.h"

namespace onnxruntime {
namespace contrib {

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcMaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearGlobalAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BeamSearch);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Attention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LongformerAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DecoderAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, EmbedLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QEmbedLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FastGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NGramRepeatBlock);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BifurcationDetector);

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Gelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Inverse);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TorchEmbedding);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Trilu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasDropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, IsAllFinite);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GridSample);

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Affine);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ParametricSoftplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ImageScaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Crop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ThresholdedRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DynamicSlice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GivenTensorFill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Scale);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRUUnit);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GivenTensorFill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Scale);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GRUUnit);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MeanVarianceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ScaledTanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Affine);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ParametricSoftplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ImageScaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Crop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DynamicSlice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ScaledTanh);

class OpSet_Microsoft_ver1 {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcMaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearGlobalAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BeamSearch)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Attention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LongformerAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DecoderAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, EmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QEmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FastGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NGramRepeatBlock)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BifurcationDetector)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Gelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Inverse)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TorchEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Trilu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasDropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, IsAllFinite)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GridSample)>());
  }
};

class OpSet_ONNX_Deprecated {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Affine)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ParametricSoftplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ImageScaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Crop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ThresholdedRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DynamicSlice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GivenTensorFill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Scale)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRUUnit)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GivenTensorFill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Scale)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GRUUnit)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MeanVarianceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ScaledTanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Affine)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ParametricSoftplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ImageScaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Crop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DynamicSlice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ScaledTanh)>());
  }
};
}  // namespace contrib
}  // namespace onnxruntime