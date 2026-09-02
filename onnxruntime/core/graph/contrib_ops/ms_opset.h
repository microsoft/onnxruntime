// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/onnx_protobuf.h"
#include "core/graph/contrib_ops/ms_schema.h"

namespace onnxruntime {
namespace contrib {
// NHWC ops
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcMaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearGlobalAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NhwcFusedConv);

// Quantization ops
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DequantizeBFP);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicQuantizeLSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicQuantizeMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulIntegerToFloat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MulInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QEmbedLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QGemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAdd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConcat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearWhere);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearLeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearSigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QuantizeBFP);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, ReduceSumInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QOrderedMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QOrderedGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QOrderedLayerNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DequantizeWithOrder);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QuantizeWithOrder);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QOrderedAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QOrderedLongformerAttention);

// Others
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Attention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BeamSearch);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, WhisperBeamSearch);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasDropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BitmaskBiasDropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasSplitGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, BiasAdd);
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
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, FusedMatMulActivation);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Gelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QuickGelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GreedySearch);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GridSample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GroupNorm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Inverse);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Irfft);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, IsAllFinite);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LongformerAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulInteger16);
#ifndef ORT_MINIMAL_BUILD
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulFpQ4);
#endif
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MaxpoolWithMask);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MoE);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QMoE);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MultiHeadAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GroupQueryAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, PagedAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LinearAttention);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, LinearAttentionGate);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatedDeltaNet);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatedRMSNorm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatedAdd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NGramHashMapping);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, VarlenNGramHashMapping);    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, EngramGate)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, CausalConvWithState)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, VarlenCausalConvWithState)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MurmurHash3)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, NGramRepeatBlock)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, PackedAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, PackedMultiHeadAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QEmbedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, RelativePositionBias)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GatedRelativePositionBias)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, RemovePadding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, RestorePadding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Rfft)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, RotaryEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MRotaryEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GemmaRotaryEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SampleOp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Sampling)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipGroupNorm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SkipSimplifiedLayerNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SparseToDenseMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, SparseAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Tokenizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TorchEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, TransposeMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Trilu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, UnfoldTensor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DynamicTimeWarping)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, Unique)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, WordConvEmbedding)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GemmFastGelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DecoderMaskedSelfAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, DecoderMaskedMultiHeadAttention)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, GemmFloat8)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulBlockQuantizedFp4Weight)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, MatMulBlockQuantizedFp8Weight)>());
  }
};
}  // namespace contrib
}  // namespace onnxruntime
