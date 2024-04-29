// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, FastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, FastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Gelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, BiasSplitGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, BiasSplitGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, QuickGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, QuickGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, QuickGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, TransposeMatMul);      // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, TransposeMatMul);     // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, TransposeMatMul);  // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, RelativePositionBias);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RelativePositionBias);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GatedRelativePositionBias);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GatedRelativePositionBias);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, RemovePadding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RemovePadding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, RestorePadding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RestorePadding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Rfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, Rfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Rfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Irfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, Irfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Irfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ComplexMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ComplexMulConj);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMulConj);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasSoftmax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasDropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BitmaskDropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BitmaskBiasDropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NGramRepeatBlock);

// These ops were experimental ops in onnx domain which have been removed now. We add them here as
// contrib ops to maintain backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Attention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Attention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, PackedAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, PackedAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, PackedMultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, PackedMultiHeadAttention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BeamSearch);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, WhisperBeamSearch);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ConvTransposeWithDynamicPads);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, Crop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, Crop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Crop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, MoE);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MoE);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QMoE);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, MultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GroupQueryAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, GroupQueryAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DecoderAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DecoderAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, DynamicSlice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int64_t, DynamicSlice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, EmbedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, EmbedLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GreedySearch);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GroupNorm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, NhwcConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, NhwcConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ImageScaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ImageScaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ImageScaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, LongformerAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, LongformerAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, RotaryEmbedding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RotaryEmbedding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, RotaryEmbedding);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GemmaRotaryEmbedding);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Sampling);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ScaledTanh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SkipGroupNorm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SkipLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SkipLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SkipSimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SkipSimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ThresholdedRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 16, float_float_float, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 16, double_double_double, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_MLFloat16, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 16, float_float_MLFloat16, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_float, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, 16, BFloat16_float_BFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float_float_float, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, double_double_double, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, float_float_MLFloat16, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16_float_float, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, BFloat16_float_BFloat16, SimplifiedLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Inverse);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MatMulNBits);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, MatMulNBits);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, MatMulBnb4);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MatMulBnb4);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, MatMulBnb4);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Trilu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, UnfoldTensor);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, DynamicTimeWarping);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int8_t, QAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int8_t, QAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, FusedConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, FastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, TransposeMatMul);  // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, FusedMatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QOrderedMatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QOrderedLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QOrderedGelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QuantizeWithOrder);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, DequantizeWithOrder);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QOrderedAttention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, QOrderedLongformerAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DecoderMaskedSelfAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DecoderMaskedSelfAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DecoderMaskedMultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DecoderMaskedMultiHeadAttention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GemmFloat8);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SparseAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, SparseAttention);

#ifdef ENABLE_ATEN
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kPytorchAtenDomain, 1, ATen);
#endif

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, ShrunkenGather);
#endif

#if defined(ORT_USE_NCCL)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, AllReduce);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, AllGather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, AllToAll);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ShardedMoE);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ShardedMoE);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedMatMul);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedSlice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedSlice);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedReshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReshape);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedExpand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedExpand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedExpand);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReduceSum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReduceSum);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReduceMax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReduceMax);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReduceMean);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReduceMean);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedUnsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedUnsqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedUnsqueeze);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedSqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DistributedSqueeze);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedSqueeze);
#endif

#ifdef ENABLE_CUDA_NHWC_OPS
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, 16, float, GridSample);
#endif

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

// Macros to avoid long line length
#define OP_CREATE_INFO(p, d, v, n) BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(p, d, v, n)>
#define OP_CREATE_INFO_TYPED(p, d, v, t, n) BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(p, d, v, t, n)>
#define OP_CREATE_INFO_VER_TYPED(p, d, vs, ve, t, n) \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(p, d, vs, ve, t, n)>

Status RegisterCudaContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, GridSample),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, FastGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, FastGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, Gelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, double, Gelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Gelu),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, BiasGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, BiasSplitGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, BiasSplitGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, BiasAdd),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, BiasAdd),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, QuickGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, double, QuickGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, QuickGelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, TransposeMatMul),      // backward compatibility
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, double, TransposeMatMul),     // backward compatibility
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, TransposeMatMul),  // backward compatibility
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, FusedMatMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, double, FusedMatMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, FusedMatMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, RelativePositionBias),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RelativePositionBias),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, GatedRelativePositionBias),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GatedRelativePositionBias),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, RemovePadding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RemovePadding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, RestorePadding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RestorePadding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, Rfft),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, double, Rfft),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Rfft),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, Irfft),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, double, Irfft),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Irfft),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, ComplexMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, ComplexMulConj),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMulConj),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, NGramRepeatBlock),

    // These ops were experimental ops in onnx domain which have been removed now. We add them here as
    // contrib ops to maintain backward compatibility
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float, Affine),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double, Affine),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Affine),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, Attention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Attention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, PackedAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, PackedAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, PackedMultiHeadAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, PackedMultiHeadAttention),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, BeamSearch),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, WhisperBeamSearch),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, ConvTransposeWithDynamicPads),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float, Crop),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double, Crop),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, Crop),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, MoE),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MoE),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QMoE),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, MultiHeadAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MultiHeadAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GroupQueryAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, GroupQueryAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DecoderAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DecoderAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, int32_t, DynamicSlice),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, int64_t, DynamicSlice),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, EmbedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, EmbedLayerNormalization),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, GreedySearch),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, GroupNorm),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, NhwcConv),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, NhwcConv),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float, ImageScaler),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double, ImageScaler),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ImageScaler),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, LongformerAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, LongformerAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float, ParametricSoftplus),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double, ParametricSoftplus),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ParametricSoftplus),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, RotaryEmbedding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, RotaryEmbedding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, RotaryEmbedding),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GemmaRotaryEmbedding),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, Sampling),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float, ScaledTanh),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double, ScaledTanh),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ScaledTanh),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, SkipGroupNorm),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, SkipLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SkipLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, SkipSimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SkipSimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float, ThresholdedRelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double, ThresholdedRelu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16, ThresholdedRelu),
    OP_CREATE_INFO_VER_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, 16, float_float_float, LayerNormalization),
    OP_CREATE_INFO_VER_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, 16, double_double_double, LayerNormalization),
    OP_CREATE_INFO_VER_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_MLFloat16, LayerNormalization),
    OP_CREATE_INFO_VER_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, 16, float_float_MLFloat16, LayerNormalization),
    OP_CREATE_INFO_VER_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_float, LayerNormalization),
    OP_CREATE_INFO_VER_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, 16, BFloat16_float_BFloat16, LayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float_float_float, SimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, double_double_double, SimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, float_float_MLFloat16, SimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, MLFloat16_float_float, SimplifiedLayerNormalization),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kOnnxDomain, 1, BFloat16_float_BFloat16, SimplifiedLayerNormalization),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, Inverse),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MatMulNBits),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, MatMulNBits),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, MatMulBnb4),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MatMulBnb4),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, MatMulBnb4),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, BiasSoftmax),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, BiasDropout),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, BitmaskDropout),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, BitmaskBiasDropout),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, QuantizeLinear),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, QuantizeLinear),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, DequantizeLinear),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, DequantizeLinear),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float_int8_t, QAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int8_t, QAttention),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, UnfoldTensor),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, DynamicTimeWarping),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, Trilu),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, FastGelu),
    // TransposedMatMul is still here for backward compatibility
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, TransposeMatMul),  // backward compatibility
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, FusedMatMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, FusedConv),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QOrderedMatMul),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QOrderedLayerNormalization),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QOrderedGelu),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QuantizeWithOrder),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, DequantizeWithOrder),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QOrderedAttention),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, QOrderedLongformerAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DecoderMaskedSelfAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DecoderMaskedSelfAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DecoderMaskedMultiHeadAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DecoderMaskedMultiHeadAttention),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, GemmFloat8),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SparseAttention),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, BFloat16, SparseAttention),

#ifdef ENABLE_ATEN
    OP_CREATE_INFO(kCudaExecutionProvider, kPytorchAtenDomain, 1, ATen),
#endif

#ifdef ENABLE_TRAINING_OPS
    // Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
    // 2). this is needed by inference for other purpose.
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, ShrunkenGather),
#endif

#if defined(ORT_USE_NCCL)
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, AllReduce),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, AllGather),
    OP_CREATE_INFO(kCudaExecutionProvider, kMSDomain, 1, AllToAll),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, ShardedMoE),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ShardedMoE),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedMatMul),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedMatMul),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedSlice),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedSlice),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedReshape),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReshape),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReshape),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedExpand),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedExpand),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedExpand),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReduceSum),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReduceSum),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReduceMax),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReduceMax),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedReduceMean),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedReduceMean),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedUnsqueeze),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedUnsqueeze),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedUnsqueeze),

    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, int64_t, DistributedSqueeze),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, float, DistributedSqueeze),
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DistributedSqueeze),
#endif

// TODO: move to RegisterCudaNhwcKernels
#ifdef ENABLE_CUDA_NHWC_OPS
    OP_CREATE_INFO_TYPED(kCudaExecutionProvider, kMSInternalNHWCDomain, 16, float, GridSample),
#endif
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

#undef OP_CREATE_INFO
#undef OP_CREATE_INFO_TYPED
#undef OP_CREATE_INFO_VER_TYPED

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
