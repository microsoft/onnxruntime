
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnx/defs/schema.h"
#include "core/graph/contrib_ops/ms_schema.h"

namespace onnxruntime {
namespace contrib {
// NHWC ops
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcMaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearGlobalAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcConv);

// Quantization ops
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicQuantizeLSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicQuantizeMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulIntegerToFloat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MulInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QEmbedLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QGemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAdd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConcat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearLeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearSigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ReduceSumInteger);

// Others
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Attention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BeamSearch);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasDropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BifurcationDetector);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, CDist);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ComplexMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ComplexMulConj);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ConvTransposeWithDynamicPads);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, CropAndResize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DecoderAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, EmbedLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ExpandDims);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FastGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedGemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Gelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GridSample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Inverse);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Irfft);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, IsAllFinite);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LongformerAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulInteger16);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MaxpoolWithMask);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MurmurHash3);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NGramRepeatBlock);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Rfft);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SampleOp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SparseToDenseMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Tokenizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TorchEmbedding);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TransposeMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Trilu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Unique);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, WordConvEmbedding);

class OpSet_Microsoft_ver1 {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcMaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearGlobalAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcConv)>());

    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DequantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicQuantizeLSTM)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicQuantizeMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulIntegerToFloat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MulInteger)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QGemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAdd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConcat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearLeakyRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearSigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ReduceSumInteger)>());

    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Attention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BeamSearch)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasDropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BifurcationDetector)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, CDist)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ComplexMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ComplexMulConj)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ConvTransposeWithDynamicPads)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, CropAndResize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DecoderAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, EmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ExpandDims)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FastGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedGemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Gelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GridSample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Inverse)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Irfft)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, IsAllFinite)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LongformerAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulInteger16)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MaxpoolWithMask)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MurmurHash3)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NGramRepeatBlock)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QEmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Rfft)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SampleOp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SparseToDenseMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Tokenizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TorchEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TransposeMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Trilu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Unique)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, WordConvEmbedding)>());
  }
};

template <typename F>
void RegisterNHWCSchema(F&& f, const ::ONNX_NAMESPACE::OpSchema& schema) {
  f(std::move(::ONNX_NAMESPACE::OpSchema(schema)
                  .TypeAndShapeInferenceFunction([](InferenceContext&) {
                    // optional. add if/when required.
                    // if we load ONNX operators we have the shape inferencing from ONNX. the conversion to the NHWC
                    // version will update those shapes. if you needed to save the model after conversion to
                    // kMSInternalNHWCDomain NHWC operators you could either
                    //   - add shape inferencing here (difficult), or
                    //   - add the type/shape info for outputs of operators in the kMSInternalNHWCDomain to
                    //     GraphProto.value_info in Graph::ToGraphProtoInternal to save the shape info with the model.
                  })
                  .SetDomain(onnxruntime::kMSInternalNHWCDomain)));
}

template <typename F>
void RegisterNHWCConvWithActivation(F&& f, const ::ONNX_NAMESPACE::OpSchema& schema) {
  f(std::move(::ONNX_NAMESPACE::OpSchema(schema)
                  .Attr("activation", "", AttributeProto::STRING, OPTIONAL_VALUE)
                  .Attr("activation_params", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
                  .TypeAndShapeInferenceFunction([](InferenceContext&) { /* optional. see comment in RegisterNHWCSchema. */ })
                  .SetDomain(onnxruntime::kMSInternalNHWCDomain)));
}

#define REGISTER_NHWC_SCHEMA(Op, SinceVersion) \
  { RegisterNHWCSchema(                        \
      fn,                                      \
      ::ONNX_NAMESPACE::GetOpSchema<           \
          ::ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, SinceVersion, Op)>()); }

#define REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(Op, SinceVersion) \
  { RegisterNHWCConvWithActivation(                            \
      fn,                                                      \
      ::ONNX_NAMESPACE::GetOpSchema<                           \
          ::ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, SinceVersion, Op)>()); }

// Schemas for ops that are NHWC versions of ONNX operators. They are created by the layout transformer by converting
// the relevant input/outputs of a node between NCHW and NHWC, and moving the node to the kMSInternalNHWCDomain domain.
class OpSet_Internal_NHWC_ver1 {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    // if the operator may be fused with an activation, use the WITH_ACTIVATION variant to add optional attributes
    // for the activation parameters.
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(Conv, 1);
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(Conv, 11);
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(MaxPool, 1);
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(MaxPool, 8);
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(MaxPool, 10);
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(MaxPool, 11);
    REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(MaxPool, 12);
    // TODO: Add other layout sensitive ops when needed. Those are:
    // QLinearConv, BatchNormalization, AveragePool, GlobalAveragePool, GlobalMaxPool, LRN, GridSample
    // DepthToSpace, SpaceToDepth
  }
};

}  // namespace contrib
}  // namespace onnxruntime
