// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/hip_common.h"
#include "core/framework/kernel_registry.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace hip {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, View);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, Group);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, SGDOptimizer);
// Adam
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_float_float, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_int64_t_float_MLFloat16_float, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_MLFloat16_float, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_float_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_int64_t_float_MLFloat16_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_MLFloat16_MLFloat16, AdamOptimizer);
// Lamb
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float_float_float_float, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float_MLFloat16_float_MLFloat16, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float_MLFloat16_float_float, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double_double_double_double_double, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_MLFloat16_MLFloat16, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_MLFloat16_float, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_float_MLFloat16, LambOptimizer);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_float_float, LambOptimizer);
// Gradient accumulator
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float, GradientAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_MLFloat16, GradientAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, ZeroGradient);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, ZeroGradient);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, SoftmaxCrossEntropy);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, SoftmaxCrossEntropyGrad);
// class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropy);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropy);
// class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropyGrad);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropyGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, BatchNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, BatchNormalizationGrad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, GatherGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, TrainableDropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, TrainableDropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, TrainableDropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, TrainableDropoutGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, TrainableDropoutGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, TrainableDropoutGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int64_t, GatherND);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int32_t, GatherND);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int64_t, GatherNDGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int32_t, GatherNDGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, DivGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, DivGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, DivGrad);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, GeluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, GeluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, GeluGrad);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, TransposeMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, TransposeMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, TransposeMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, IsFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, IsFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, IsFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, bool, All);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, IsAllFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, IsAllFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, IsAllFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, MixedPrecisionScale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, MixedPrecisionScale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_MLFloat16, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_MLFloat16, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float, LayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double_float, LayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float, LayerNormalizationGrad);

#ifdef USE_HOROVOD
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, HorovodAllReduce);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, HorovodBarrier);

// P2P communication operators.
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, Send);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, Send);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, Send);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, Recv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, Recv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, Recv);
#endif
#ifdef USE_NCCL
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, NcclAllReduce);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, NcclAllGather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, NcclReduceScatter);
#endif

void RegisterHipTrainingKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, View)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, Group)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, SGDOptimizer)>,

      // Adam
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_float_float, AdamOptimizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_int64_t_float_MLFloat16_float, AdamOptimizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_MLFloat16_float, AdamOptimizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_float_MLFloat16, AdamOptimizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_int64_t_float_MLFloat16_MLFloat16, AdamOptimizer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_int64_t_float_MLFloat16_MLFloat16, AdamOptimizer)>,

      // Lamb
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float_float_float_float, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float_MLFloat16_float_MLFloat16, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float_MLFloat16_float_float, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double_double_double_double_double, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_MLFloat16_MLFloat16, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_MLFloat16_float, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_float_MLFloat16, LambOptimizer)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float_MLFloat16_float_float, LambOptimizer)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float, GradientAccumulator)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_MLFloat16, GradientAccumulator)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, ZeroGradient)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, ZeroGradient)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, TrainableDropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, TrainableDropout)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, TrainableDropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, TrainableDropoutGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, TrainableDropoutGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, TrainableDropoutGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int64_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int32_t, GatherND)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int64_t, GatherNDGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, int32_t, GatherNDGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, SoftmaxCrossEntropy)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, SoftmaxCrossEntropyGrad)>,
      // // BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropy)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropy)>,
      // // BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropyGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropyGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, SoftmaxGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, SoftmaxGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, SoftmaxGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, BatchNormalizationGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, BatchNormalizationGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, GatherGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, DivGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, DivGrad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, DivGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, GeluGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, GeluGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, GeluGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, TransposeMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, TransposeMatMul)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, TransposeMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, IsFinite)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, IsFinite)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, IsFinite)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, bool, All)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, IsAllFinite)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, IsAllFinite)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, IsAllFinite)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, MixedPrecisionScale)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, MixedPrecisionScale)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float, ReduceAllL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float, ReduceAllL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_MLFloat16, ReduceAllL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_MLFloat16, ReduceAllL2)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float_float, LayerNormalizationGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double_float, LayerNormalizationGrad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16_float, LayerNormalizationGrad)>,

#ifdef USE_HOROVOD
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, HorovodAllReduce)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, HorovodBarrier)>,
      // P2P communication operators.
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, Send)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, Send)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, Send)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, float, Recv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, MLFloat16, Recv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kMSDomain, 1, double, Recv)>,

#endif
#ifdef USE_NCCL
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, NcclAllReduce)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, NcclAllGather)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, NcclReduceScatter)>,
#endif
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

}  // namespace hip
}  // namespace onnxruntime
