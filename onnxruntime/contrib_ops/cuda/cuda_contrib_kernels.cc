// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::common;

// Macros to avoid long line length
#define CUDA_MS_OP_CLASS_NAME(ver, name) \
  ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, ver, name)
#define CUDA_MS_OP_TYPED_CLASS_NAME(ver, type, name) \
  ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, ver, type, name)
#define CUDA_MS_OP_VERSIONED_TYPED_CLASS_NAME(start_ver, end_ver, type, name) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, start_ver, end_ver, type, name)
#define CUDA_MS_OP_VERSIONED_CLASS_NAME(start_ver, end_ver, name) \
  ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, start_ver, end_ver, name)

#define CUDA_ONNX_OP_TYPED_CLASS_NAME(ver, type, name) \
  ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, ver, type, name)
#define CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(start_ver, end_ver, type, name) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, start_ver, end_ver, type, name)

namespace onnxruntime {
namespace contrib {
namespace cuda {
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GridSample);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Gelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Gelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Gelu);
class CUDA_MS_OP_CLASS_NAME(1, BiasGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasSplitGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasSplitGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, TransposeMatMul);      // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, TransposeMatMul);     // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, TransposeMatMul);  // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, FusedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FusedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedRelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedRelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RemovePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RemovePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RestorePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RestorePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Rfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Rfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Rfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Irfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Irfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Irfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMulConj);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMulConj);
class CUDA_MS_OP_CLASS_NAME(1, BiasSoftmax);
class CUDA_MS_OP_CLASS_NAME(1, BiasDropout);
class CUDA_MS_OP_CLASS_NAME(1, BitmaskDropout);
class CUDA_MS_OP_CLASS_NAME(1, BitmaskBiasDropout);
class CUDA_MS_OP_CLASS_NAME(1, NGramRepeatBlock);

// These ops were experimental ops in onnx domain which have been removed now. We add them here as
// contrib ops to maintain backward compatibility
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Affine);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Affine);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Affine);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Attention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Attention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedMultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedMultiHeadAttention);
class CUDA_MS_OP_CLASS_NAME(1,  PagedAttention);
class CUDA_MS_OP_CLASS_NAME(1, BeamSearch);
class CUDA_MS_OP_CLASS_NAME(1, WhisperBeamSearch);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ConvTransposeWithDynamicPads);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Crop);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Crop);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Crop);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MoE);
class CUDA_MS_OP_CLASS_NAME(1, QMoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderAttention);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int32_t, DynamicSlice);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int64_t, DynamicSlice);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, EmbedLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, EmbedLayerNormalization);
class CUDA_MS_OP_CLASS_NAME(1, GreedySearch);
class CUDA_MS_OP_CLASS_NAME(1, GroupNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, NhwcConv);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, NhwcConv);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ImageScaler);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ImageScaler);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ImageScaler);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LongformerAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LongformerAttention);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ParametricSoftplus);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ParametricSoftplus);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ParametricSoftplus);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, RotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GemmaRotaryEmbedding);
class CUDA_MS_OP_CLASS_NAME(1, Sampling);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ScaledTanh);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ScaledTanh);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ScaledTanh);
class CUDA_MS_OP_CLASS_NAME(1, SkipGroupNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipSimplifiedLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipSimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ThresholdedRelu);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ThresholdedRelu);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ThresholdedRelu);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_float, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, double_double_double, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_MLFloat16, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_MLFloat16, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_float, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, BFloat16_float_BFloat16, LayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_float, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double_double_double, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_MLFloat16, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_float, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, BFloat16_float_BFloat16, SimplifiedLayerNormalization);
class CUDA_MS_OP_CLASS_NAME(1, Inverse);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulNBits);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulNBits);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MatMulBnb4);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulBnb4);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulBnb4);
class CUDA_MS_OP_CLASS_NAME(1, Trilu);
class CUDA_MS_OP_CLASS_NAME(1, UnfoldTensor);
class CUDA_MS_OP_CLASS_NAME(1, DynamicTimeWarping);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, QuantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, QuantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, DequantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, DequantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_int8_t, QAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, QAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedConv);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, TransposeMatMul);  // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FusedMatMul);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedMatMul);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedLayerNormalization);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedGelu);
class CUDA_MS_OP_CLASS_NAME(1, QuantizeWithOrder);
class CUDA_MS_OP_CLASS_NAME(1, DequantizeWithOrder);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedAttention);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedLongformerAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderMaskedSelfAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderMaskedSelfAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderMaskedMultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderMaskedMultiHeadAttention);
class CUDA_MS_OP_CLASS_NAME(1, GemmFloat8);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SparseAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SparseAttention);

#ifdef ENABLE_ATEN
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kPytorchAtenDomain, 1, ATen);
#endif

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once
// 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.
class CUDA_MS_OP_CLASS_NAME(1, ShrunkenGather);
#endif

#if defined(ORT_USE_NCCL)
class CUDA_MS_OP_CLASS_NAME(1, AllReduce);
class CUDA_MS_OP_CLASS_NAME(1, AllGather);
class CUDA_MS_OP_CLASS_NAME(1, AllToAll);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ShardedMoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ShardedMoE);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedMatMul);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSlice);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSlice);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedReshape);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReshape);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReshape);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedExpand);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedExpand);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedExpand);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceSum);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceSum);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMax);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMax);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMean);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMean);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedUnsqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedUnsqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedUnsqueeze);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedSqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSqueeze);
#endif

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterCudaContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GridSample)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FastGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FastGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BiasGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasSplitGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasSplitGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, TransposeMatMul)>,      // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, TransposeMatMul)>,     // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, TransposeMatMul)>,  // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedRelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedRelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RemovePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RemovePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RestorePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RestorePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Rfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Rfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Rfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Irfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Irfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Irfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMulConj)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMulConj)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, NGramRepeatBlock)>,

      // These ops were experimental ops in onnx domain which have been removed now. We add them here as
      // contrib ops to maintain backward compatibility
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Affine)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Affine)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Affine)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Attention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Attention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, PagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BeamSearch)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, WhisperBeamSearch)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ConvTransposeWithDynamicPads)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Crop)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Crop)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Crop)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QMoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderAttention)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int32_t, DynamicSlice)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int64_t, DynamicSlice)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, EmbedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, EmbedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, GreedySearch)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, GroupNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, NhwcConv)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, NhwcConv)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ImageScaler)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ImageScaler)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ImageScaler)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LongformerAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LongformerAttention)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ParametricSoftplus)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ParametricSoftplus)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ParametricSoftplus)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, RotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GemmaRotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, Sampling)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ScaledTanh)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ScaledTanh)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ScaledTanh)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, SkipGroupNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipSimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipSimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ThresholdedRelu)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ThresholdedRelu)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ThresholdedRelu)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_float, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, double_double_double, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_MLFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_MLFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_float, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, BFloat16_float_BFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_float, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double_double_double, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_MLFloat16, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_float, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, BFloat16_float_BFloat16, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, Inverse)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulNBits)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulNBits)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MatMulBnb4)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulBnb4)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulBnb4)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BiasSoftmax)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BiasDropout)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BitmaskDropout)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BitmaskBiasDropout)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_int8_t, QAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, QAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, UnfoldTensor)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, DynamicTimeWarping)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, Trilu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FastGelu)>,
      // TransposedMatMul is still here for backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, TransposeMatMul)>,  // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedConv)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QuantizeWithOrder)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, DequantizeWithOrder)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedLongformerAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderMaskedSelfAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderMaskedSelfAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderMaskedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderMaskedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, GemmFloat8)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SparseAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SparseAttention)>,

#ifdef ENABLE_ATEN
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kPytorchAtenDomain, 1, ATen)>,
#endif

#ifdef ENABLE_TRAINING_OPS
      // Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once
      // 1). compute optimizer is enabled for inference or
      // 2). this is needed by inference for other purpose.
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, ShrunkenGather)>,
#endif

#if defined(ORT_USE_NCCL)
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, AllReduce)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, AllGather)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, AllToAll)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ShardedMoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ShardedMoE)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedMatMul)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSlice)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSlice)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedReshape)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReshape)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReshape)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedExpand)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedExpand)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedExpand)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceSum)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceSum)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMax)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMax)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMean)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMean)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedUnsqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedUnsqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedUnsqueeze)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedSqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSqueeze)>,
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
