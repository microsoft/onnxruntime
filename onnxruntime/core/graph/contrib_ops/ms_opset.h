// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif
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


class OpSet_Microsoft_ver1 {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, NhwcMaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, QLinearGlobalAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, QLinearAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, QLinearConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, BeamSearch)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, Attention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, QAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, LongformerAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, DecoderAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, EmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, QEmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, FastGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, SkipLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, NGramRepeatBlock)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, BifurcationDetector)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, Gelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, BiasGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, Inverse)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, TorchEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, Trilu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, BiasSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, BiasDropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, IsAllFinite)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Microsoft, 1, GridSample)>());
  }
};
}  // namespace contrib
}  // namespace onnxruntime