// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_common.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace rocm {
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, GridSample);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, FastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, FastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, Gelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Gelu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BiasGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, BiasSplitGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, BiasSplitGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, BiasAdd);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, QuickGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, QuickGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, QuickGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, TransposeMatMul);      // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, TransposeMatMul);     // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, TransposeMatMul);  // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Rfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, Rfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Rfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Irfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, Irfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Irfft);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, ComplexMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, ComplexMulConj);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMulConj);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BiasSoftmax);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BiasDropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BitmaskDropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BitmaskBiasDropout);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, NGramRepeatBlock);

// These ops were experimental ops in onnx domain which have been removed now. We add them here as
// contrib ops to maintain backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, Affine);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Attention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Attention);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BeamSearch);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, ConvTransposeWithDynamicPads);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, Crop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, Crop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, Crop);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, MultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, MultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, DecoderAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, DecoderAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, DecoderMaskedMultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, DecoderMaskedMultiHeadAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, int32_t, DynamicSlice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, int64_t, DynamicSlice);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, EmbedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, EmbedLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, GroupNorm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, NhwcConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, NhwcConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ImageScaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ImageScaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ImageScaler);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, LongformerAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, LongformerAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ParametricSoftplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ParametricSoftplus);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, Sampling);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ScaledTanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, SkipLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, SkipLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, SkipSimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, SkipSimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ThresholdedRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ThresholdedRelu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, float_float_float, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, double_double_double, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_MLFloat16, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, float_float_MLFloat16, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_float, LayerNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, BFloat16_float_BFloat16, LayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float_float_float, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double_double_double, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float_float_MLFloat16, SimplifiedLayerNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16_float_float, SimplifiedLayerNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, Inverse);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, Trilu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, QuantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, DequantizeLinear);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float_int8_t, QAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16_int8_t, QAttention);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, FusedConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, FusedConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, FastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, TransposeMatMul);  // backward compatibility
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, GemmFastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, GemmFastGelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, GemmFastGelu);

#ifdef ENABLE_ATEN
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kPytorchAtenDomain, 1, ATen);
#endif

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, ShrunkenGather);
#endif

#if defined(USE_MPI) && defined(ORT_USE_NCCL)
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, AllReduce);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, AllGather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, AllToAll);
#endif

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterRocmContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, GridSample)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, FastGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, FastGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Gelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, Gelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Gelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BiasGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, BiasSplitGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, BiasSplitGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, BiasAdd)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, BiasAdd)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, QuickGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, QuickGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, QuickGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, TransposeMatMul)>,      // backward compatibility
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, TransposeMatMul)>,     // backward compatibility
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, TransposeMatMul)>,  // backward compatibility
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, FusedMatMul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, FusedMatMul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, FusedMatMul)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Rfft)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, Rfft)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Rfft)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Irfft)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, double, Irfft)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Irfft)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, ComplexMul)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMul)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, ComplexMulConj)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, ComplexMulConj)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                          1, NGramRepeatBlock)>,

    // These ops were experimental ops in onnx domain which have been removed now. We add them here as
    // contrib ops to maintain backward compatibility
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, Affine)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, Affine)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, Affine)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, Attention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, Attention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BeamSearch)>,

    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, ConvTransposeWithDynamicPads)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, Crop)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, Crop)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, Crop)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, MultiHeadAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, MultiHeadAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                                1, float, DecoderAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                                1, MLFloat16, DecoderAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, DecoderMaskedMultiHeadAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, DecoderMaskedMultiHeadAttention)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, int32_t, DynamicSlice)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, int64_t, DynamicSlice)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, EmbedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, EmbedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, GroupNorm)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, NhwcConv)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, NhwcConv)>,

    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ImageScaler)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ImageScaler)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ImageScaler)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                                1, float, LongformerAttention)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                                1, MLFloat16, LongformerAttention)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ParametricSoftplus)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ParametricSoftplus)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ParametricSoftplus)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, Sampling)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ScaledTanh)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ScaledTanh)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ScaledTanh)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, SkipLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, SkipLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, SkipSimplifiedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, SkipSimplifiedLayerNormalization)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float, ThresholdedRelu)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double, ThresholdedRelu)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16, ThresholdedRelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, float_float_float, LayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, double_double_double, LayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_MLFloat16, LayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, float_float_MLFloat16, LayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, MLFloat16_float_float, LayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, 16, BFloat16_float_BFloat16, LayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float_float_float, SimplifiedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, double_double_double, SimplifiedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, float_float_MLFloat16, SimplifiedLayerNormalization)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kOnnxDomain, 1, MLFloat16_float_float, SimplifiedLayerNormalization)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, Inverse)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, Trilu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BiasSoftmax)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BiasDropout)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BitmaskDropout)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BitmaskBiasDropout)>,

    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, QuantizeLinear)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, QuantizeLinear)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, int8_t_MLFloat16, DequantizeLinear)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, uint8_t_MLFloat16, DequantizeLinear)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float_int8_t, QAttention)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16_int8_t, QAttention)>

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, FastGelu)>,
    // TransposedMatMul is still here for backward compatibility
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, TransposeMatMul)>,  // backward compatibility
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, FusedMatMul)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                                1, float, FusedConv)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain,
                                                                1, MLFloat16, FusedConv)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, float, GemmFastGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, MLFloat16, GemmFastGelu)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, BFloat16, GemmFastGelu)>,

#ifdef ENABLE_ATEN
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kPytorchAtenDomain, 1, ATen)>,
#endif

#ifdef ENABLE_TRAINING_OPS
    // Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
    // 2). this is needed by inference for other purpose.
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, ShrunkenGather)>,
#endif

#if defined(USE_MPI) && defined(ORT_USE_NCCL)
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, AllReduce)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, AllGather)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kRocmExecutionProvider, kMSDomain, 1, AllToAll)>,
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

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
