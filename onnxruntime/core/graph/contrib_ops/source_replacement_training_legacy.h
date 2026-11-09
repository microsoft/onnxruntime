// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/contrib_ops/source_replacement_policy.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
#include <array>

namespace onnxruntime::contrib {

// Separately frozen optional training schema inventory from base
// 63dcf15dfd50f2cc4941a45bb01c210325d88cd0. The actual training_op_defs.cc
// (git blob 92f803030ada49b3a09b4411aff7a61608c9452b) was compiled and registered
// in a task-owned Linux x86_64 Release schema-only probe with ENABLE_TRAINING_OPS
// and ENABLE_TRITON, using the exact prechange contrib registration macro.
// No training/Triton/NCCL kernels were compiled or qualified by this inventory.
// 101 sorted exact keys; canonical JSON tuple-list SHA256:
// bd0bf249c1a43a69d7d1fff4c74f81316273596bc13537cfb9b66e36327b276a
// Do not expand this baseline to exempt newly authored schemas or versions.
inline gsl::span<const OpKey> LegacyTrainingSourcePolicyKeys() {
  static const std::array<OpKey, 101> keys{{
      {"", "AveragePoolGrad", 9},
      {"", "GatherND", 1},
      {"", "MaxPoolGrad", 9},
      {"", "SparseSoftmaxCrossEntropy", 9},
      {"", "SparseSoftmaxCrossEntropyGrad", 9},
      {"com.microsoft", "AdamOptimizer", 1},
      {"com.microsoft", "AdamWOptimizer", 1},
      {"com.microsoft", "AdasumAllReduce", 1},
      {"com.microsoft", "All", 1},
      {"com.microsoft", "BatchNormInternal", 1},
      {"com.microsoft", "BatchNormalizationGrad", 1},
      {"com.microsoft", "BatchScale", 1},
      {"com.microsoft", "BiasFastGeluGrad_dX", 1},
      {"com.microsoft", "BiasGeluGrad_dX", 1},
      {"com.microsoft", "BiasSoftmaxDropout", 1},
      {"com.microsoft", "BitmaskDropoutGrad", 1},
      {"com.microsoft", "BroadcastGradientArgs", 1},
      {"com.microsoft", "ConcatTraining", 1},
      {"com.microsoft", "ConvGrad", 1},
      {"com.microsoft", "ConvTransposeGrad", 1},
      {"com.microsoft", "DivGrad", 1},
      {"com.microsoft", "DropoutGrad", 1},
      {"com.microsoft", "FakeQuant", 1},
      {"com.microsoft", "FakeQuantGrad", 1},
      {"com.microsoft", "FastGeluGrad", 1},
      {"com.microsoft", "FlattenAndUnpad", 1},
      {"com.microsoft", "GRUGrad", 1},
      {"com.microsoft", "GRUTraining", 1},
      {"com.microsoft", "GatherElementsGrad", 1},
      {"com.microsoft", "GatherGrad", 1},
      {"com.microsoft", "GatherNDGrad", 1},
      {"com.microsoft", "GeluGrad", 1},
      {"com.microsoft", "GistBinarizeDecoder", 1},
      {"com.microsoft", "GistBinarizeEncoder", 1},
      {"com.microsoft", "GistPack16Decoder", 1},
      {"com.microsoft", "GistPack16Encoder", 1},
      {"com.microsoft", "GistPack1Decoder", 1},
      {"com.microsoft", "GistPack1Encoder", 1},
      {"com.microsoft", "GistPack8Decoder", 1},
      {"com.microsoft", "GistPack8Encoder", 1},
      {"com.microsoft", "GistPackMsfp15Decoder", 1},
      {"com.microsoft", "GistPackMsfp15Encoder", 1},
      {"com.microsoft", "Group", 1},
      {"com.microsoft", "InPlaceAccumulator", 1},
      {"com.microsoft", "InPlaceAccumulatorV2", 1},
      {"com.microsoft", "InplaceClipGradNorm", 1},
      {"com.microsoft", "InvertibleLayerNormalizationGrad", 1},
      {"com.microsoft", "IsFinite", 1},
      {"com.microsoft", "LSTMGrad", 1},
      {"com.microsoft", "LSTMTraining", 1},
      {"com.microsoft", "LambOptimizer", 1},
      {"com.microsoft", "LayerNormalizationGrad", 1},
      {"com.microsoft", "LeakyReluGrad", 1},
      {"com.microsoft", "LogSoftmaxGrad", 1},
      {"com.microsoft", "LogSoftmaxGrad_13", 1},
      {"com.microsoft", "MegatronF", 1},
      {"com.microsoft", "MegatronG", 1},
      {"com.microsoft", "MixedPrecisionScale", 1},
      {"com.microsoft", "NcclAllGather", 1},
      {"com.microsoft", "NcclAllReduce", 1},
      {"com.microsoft", "NcclReduceScatter", 1},
      {"com.microsoft", "NegativeLogLikelihoodLossInternal", 1},
      {"com.microsoft", "NegativeLogLikelihoodLossInternal2", 1},
      {"com.microsoft", "PadAndUnflatten", 1},
      {"com.microsoft", "PassThrough", 1},
      {"com.microsoft", "PythonOp", 1},
      {"com.microsoft", "PythonOpGrad", 1},
      {"com.microsoft", "QuickGeluGrad", 1},
      {"com.microsoft", "RecordEvent", 1},
      {"com.microsoft", "Recv", 1},
      {"com.microsoft", "ReduceAllL2", 1},
      {"com.microsoft", "ReduceSumTraining", 1},
      {"com.microsoft", "ReluGrad", 1},
      {"com.microsoft", "ResizeGrad", 1},
      {"com.microsoft", "SGDOptimizer", 1},
      {"com.microsoft", "SGDOptimizerV2", 1},
      {"com.microsoft", "Scale", 1},
      {"com.microsoft", "ScaledSum", 1},
      {"com.microsoft", "Send", 1},
      {"com.microsoft", "SigmoidGrad", 1},
      {"com.microsoft", "SimplifiedLayerNormalizationGrad", 1},
      {"com.microsoft", "SliceGrad", 1},
      {"com.microsoft", "SoftmaxCrossEntropy", 1},
      {"com.microsoft", "SoftmaxCrossEntropyGrad", 1},
      {"com.microsoft", "SoftmaxCrossEntropyLossGrad", 1},
      {"com.microsoft", "SoftmaxCrossEntropyLossInternal", 1},
      {"com.microsoft", "SoftmaxCrossEntropyLossInternalGrad", 1},
      {"com.microsoft", "SoftmaxDropoutGrad", 1},
      {"com.microsoft", "SoftmaxGrad", 1},
      {"com.microsoft", "SoftmaxGrad_13", 1},
      {"com.microsoft", "SplitTraining", 1},
      {"com.microsoft", "SummaryHistogram", 1},
      {"com.microsoft", "SummaryMerge", 1},
      {"com.microsoft", "SummaryScalar", 1},
      {"com.microsoft", "SummaryText", 1},
      {"com.microsoft", "TanhGrad", 1},
      {"com.microsoft", "TritonOp", 1},
      {"com.microsoft", "View", 1},
      {"com.microsoft", "WaitEvent", 1},
      {"com.microsoft", "YieldOp", 1},
      {"com.microsoft", "ZeroGradient", 1},
  }};
  return keys;
}
}  // namespace onnxruntime::contrib
#endif
