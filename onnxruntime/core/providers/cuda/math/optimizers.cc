// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optimizers.h"
#include "binary_elementwise_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .Alias(1, 0)  // Update weights in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer);

Status SGDOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor& NW = *ctx->Output(0, W.Shape());

  ORT_ENFORCE(W.Shape() == G.Shape());

  SGDOptimizerImpl(
      ETA.template Data<float>(),
      W.template Data<float>(),
      G.template Data<float>(),
      NW.template MutableData<float>(),
      W.Shape().Size());

  return Status::OK();
}

template <typename T>
Status CopyIfNotSameBuffer(const Tensor& source_tensor, Tensor& target_tensor) {
  const T* source = source_tensor.template Data<T>();
  T* target = target_tensor.template MutableData<T>();
  if (target != source) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, source_tensor.SizeInBytes(), cudaMemcpyDeviceToDevice));
  }
  return Status::OK();
}

// TODO: Once Schema is checked in to onnx lets fix this to match that
#define REGISTER_ADAM_KERNEL_TYPED(T1, T2, T3, T4, T_GRAD)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                      \
      AdamOptimizer,                                                                  \
      kOnnxDomain,                                                                    \
      9,                                                                              \
      T1##_##T2##_##T3##_##T4##_##T_GRAD,                                             \
      kCudaExecutionProvider,                                                         \
      KernelDefBuilder()                                                              \
          .Alias(1, 3)                             /* Update step count in-place */   \
          .Alias(2, 0)                             /* Update weights in-place */      \
          .Alias(4, 1)                             /* Update moment-1 in-place */     \
          .Alias(5, 2)                             /* Update moment-2 in-place */     \
          .Alias(6, 4)                             /* Update FP16 weights in-place */ \
          .InputMemoryType<OrtMemTypeCPUInput>(1)  /* Keep step count in CPU */       \
          .InputMemoryType<OrtMemTypeCPUInput>(8)  /* Keep do_update in CPU */        \
          .OutputMemoryType<OrtMemTypeCPUInput>(3) /* Keep step count in CPU */       \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                    \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())                    \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())                    \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>())            \
          .TypeConstraint("T_FP16", DataTypeImpl::GetTensorType<MLFloat16>())         \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),                  \
      AdamOptimizer<T1, T2, T3, T4, T_GRAD>);

REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, float)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(MLFloat16, int64_t, float, MLFloat16, MLFloat16)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, MLFloat16, MLFloat16)

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
Status AdamOptimizer<T1, T2, T3, T4, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NW = *ctx->Output(0, W.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor& NS = *ctx->Output(3, S.Shape());

  half* fp16_weights_out = nullptr;
  if (ctx->InputCount() >= 7 && ctx->OutputCount() >= 5) {
    const Tensor& W_FP16 = *ctx->Input<Tensor>(6);
    Tensor& NW_FP16 = *ctx->Output(4, W_FP16.Shape());
    fp16_weights_out = reinterpret_cast<half*>(NW_FP16.template MutableData<MLFloat16>());
  }

  const CudaT3* loss_scale = nullptr;
  if (ctx->InputCount() >= 8) {
    const Tensor& loss_scale_tensor = *ctx->Input<Tensor>(7);
    loss_scale = reinterpret_cast<const CudaT3*>(loss_scale_tensor.template Data<T3>());
  }

  const T2* S_in = S.template Data<T2>();
  if (ctx->InputCount() >= 9) {
    const Tensor& do_update_tensor = *ctx->Input<Tensor>(8);
    const bool do_update = *do_update_tensor.template Data<bool>();
    if (!do_update) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T3>(W, NW));
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(M1, NM1));
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(M2, NM2));
      if (S_in != NS.template MutableData<T2>()) {
        *(NS.template MutableData<T2>()) = *(S_in);
      }

      if (fp16_weights_out) {
        const Tensor& W_FP16 = *ctx->Input<Tensor>(6);
        Tensor& NW_FP16 = *ctx->Output(4, W_FP16.Shape());
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<MLFloat16>(W_FP16, NW_FP16));
      }
      return Status::OK();
    }
  }

  AdamOptimizerImpl(
      reinterpret_cast<const CudaT1*>(ETA.template Data<T1>()),
      *S_in,
      reinterpret_cast<const CudaT3*>(W.template Data<T3>()),
      reinterpret_cast<const CudaT_GRAD*>(G.template Data<T_GRAD>()),
      reinterpret_cast<const CudaT4*>(M1.template Data<T4>()),
      reinterpret_cast<const CudaT4*>(M2.template Data<T4>()),
      loss_scale,
      ToCudaType<T4>::FromFloat(alpha_),
      ToCudaType<T4>::FromFloat(beta_),
      ToCudaType<T4>::FromFloat(lambda_),
      ToCudaType<T4>::FromFloat(epsilon_),
      reinterpret_cast<CudaT3*>(NW.template MutableData<T3>()),
      reinterpret_cast<CudaT4*>(NM1.template MutableData<T4>()),
      reinterpret_cast<CudaT4*>(NM2.template MutableData<T4>()),
      fp16_weights_out,
      W.Shape().Size());

  *(NS.template MutableData<T2>()) = *(S_in) + 1;

  return Status::OK();
}

std::vector<std::pair<int, int>> GenerateLambExtraAliasMapping() {
  // Starting index of extra inputs.
  constexpr int input_index_bias = 3;
  // Starting index of extra outputs.
  constexpr int output_index_bias = 0;
  // Count of extra I/O groups. One group corresponds to a weight update.
  constexpr int group_count = 1024;
  // length of [w, g, m1, m2, w_fp16].
  constexpr int input_stride = 5;
  // length of [w_new, m1_new, m2_new, w_fp16_new].
  constexpr int output_stride = 4;

  std::vector<std::pair<int, int>> alias_pairs{};
  for (int i = 0; i < group_count; ++i) {
    const int input = input_index_bias + i * input_stride;
    const int output = output_index_bias + i * output_stride;
    // w --> w_new
    alias_pairs.emplace_back(std::make_pair(input, output));
    // m1 --> m1_new
    alias_pairs.emplace_back(std::make_pair(input + 2, output + 1));
    // m2 --> m2_new
    alias_pairs.emplace_back(std::make_pair(input + 3, output + 2));
    // w_fp16 --> w_fp16_new
    alias_pairs.emplace_back(std::make_pair(input + 4, output + 3));
  }

  return alias_pairs;
}

// TODO: Once Schema is checked in to onnx lets fix this to match that
#define REGISTER_LAMB_KERNEL_TYPED(T1, T2, T3, T4)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                     \
      LambOptimizer,                                                                 \
      kOnnxDomain,                                                                   \
      9,                                                                             \
      T1##_##T2##_##T3##_##T4,                                                       \
      kCudaExecutionProvider,                                                        \
      KernelDefBuilder()                                                             \
          .Alias(GenerateLambExtraAliasMapping())                                    \
          .InputMemoryType<OrtMemTypeCPUInput>(1) /* Keep do_update in CPU */        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                   \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())                   \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())                   \
          .TypeConstraint("T_FP16", DataTypeImpl::GetTensorType<MLFloat16>())        \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),                 \
      LambOptimizer<T1, T2, T3, T4>);

REGISTER_LAMB_KERNEL_TYPED(float, float, MLFloat16, float)
REGISTER_LAMB_KERNEL_TYPED(float, float, float, float)
REGISTER_LAMB_KERNEL_TYPED(double, double, double, double)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, float)

template <typename T1, typename T2, typename T3, typename T4>
Status LambOptimizer<T1, T2, T3, T4>::ComputeInternal(OpKernelContext* ctx) const {
  // CudaT* are types used to invoke CUDA-based functions. It, for example, maps
  // MLFloat16 in ONNXRuntime to half in CUDA.
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;

  constexpr int non_grouped_input_count = 3;
  constexpr int input_group_size = 5;
  constexpr int output_group_size = 4;
  constexpr int minimal_input_count = non_grouped_input_count + 1 * input_group_size - 1;
  constexpr int minimal_output_count = 1 * output_group_size - 1;
  const int grouped_input_tensor_count = ctx->InputCount() - non_grouped_input_count;
  const int grouped_output_tensor_count = ctx->OutputCount();

  // At least one variable group for updating one weight tensor.
  ORT_ENFORCE(
    ctx->InputCount() >= minimal_input_count,
    "Expect at least ", minimal_input_count, " inputs but got ",
    ctx->InputCount());
  // At least one variable group for updating one weight tensor.
  ORT_ENFORCE(
    ctx->OutputCount() >= minimal_output_count,
    "Expect at least ", minimal_output_count, " outputs but got ",
    ctx->OutputCount());

  // In addition to the first 3 inputs, all inputs are repeated sequence of [w, g, m1, m2, w_fp16].
  ORT_ENFORCE(
    grouped_input_tensor_count % input_group_size == 0,
    "Input count must be ", non_grouped_input_count, " + ", input_group_size,
    " x (numer of weights to optimize).");
  // Outputs are repeated sequence of [w_new, m1_new, m2_new, w_fp16_new].
  ORT_ENFORCE(
    grouped_output_tensor_count % output_group_size == 0,
    "Output count must be ", output_group_size,
    " x (numer of weights to optimize).");
  // Number of repeated [w, g, m1, m2, w_fp16]'s should match number of repeated [w_new, m1_new, m2_new, w_fp16_new].
  ORT_ENFORCE(
    grouped_input_tensor_count / input_group_size == grouped_output_tensor_count / output_group_size,
    "Input and output tensor counts are not aligned. Please check LambOptimizer's input and output lists.");

  // Number of [w, g, m1, m2, (w_fp16)] (or [w_new, m1_new, m2_new, (w_fp16_new)]).
  const int group_count = (grouped_input_tensor_count + input_group_size - 1) / input_group_size;

  // At least we need one group of alpha, beta, lambda, ..., for processing one group.
  ORT_ENFORCE(alpha_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(beta_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(lambda_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(epsilon_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(threshold_.size() >= static_cast<size_t>(group_count));

  if (ctx->Input<Tensor>(1)) {
    const Tensor& update_signal_tensor = *ctx->Input<Tensor>(1);
    const bool update_signal = *update_signal_tensor.template Data<bool>();
    if (!update_signal) {
      for (int group_index = 0; group_index < group_count; ++group_index) {
        const int input_start_index = non_grouped_input_count + group_index * input_group_size;
        const Tensor& w = *ctx->Input<Tensor>(input_start_index);
        const Tensor& m1 = *ctx->Input<Tensor>(input_start_index + 2);
        const Tensor& m2 = *ctx->Input<Tensor>(input_start_index + 3);
        const int output_start_index = group_index * output_group_size;
        Tensor& w_new = *ctx->Output(output_start_index, w.Shape());
        Tensor& m1_new = *ctx->Output(output_start_index + 1, w.Shape());
        Tensor& m2_new = *ctx->Output(output_start_index + 2, w.Shape());

        ORT_ENFORCE(w.Shape() == m1.Shape());
        ORT_ENFORCE(w.Shape() == m2.Shape());
        ORT_ENFORCE(w.Shape() == m1_new.Shape());
        ORT_ENFORCE(w.Shape() == m2_new.Shape());

        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T2>(w, w_new));
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(m1, m1_new));
        ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T4>(m2, m2_new));

        const Tensor* w_fp16 = ctx->Input<Tensor>(input_start_index + 4);
        Tensor* w_fp16_new = ctx->Output(output_start_index + 3, w.Shape());
        if (w_fp16 && w_fp16_new) {
          ORT_ENFORCE(w.Shape() == w_fp16->Shape());
          ORT_ENFORCE(w.Shape() == w_fp16_new->Shape());
          ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<MLFloat16>(*w_fp16, *w_fp16_new));
        }
      }
      return Status::OK();
    }
  }

  const CudaT2* loss_scale_data = nullptr;
  if (ctx->Input<Tensor>(0)) {
    const Tensor& loss_scale_tensor = *ctx->Input<Tensor>(0);
    loss_scale_data = reinterpret_cast<const CudaT2*>(loss_scale_tensor.template Data<T2>());
  }

  const Tensor& eta = *ctx->Input<Tensor>(2);
  const CudaT1* eta_data = reinterpret_cast<const CudaT1*>(eta.template Data<T1>());

  for (int group_index = 0; group_index < group_count; ++group_index) {
    // Prepare used input tensors for this group.
    const int input_start_index = non_grouped_input_count + group_index * input_group_size;
    const Tensor& w = *ctx->Input<Tensor>(input_start_index);
    const Tensor& g = *ctx->Input<Tensor>(input_start_index + 1);
    const Tensor& m1 = *ctx->Input<Tensor>(input_start_index + 2);
    const Tensor& m2 = *ctx->Input<Tensor>(input_start_index + 3);

    // Prepare used outputs tensors for this group.
    const int output_start_index = group_index * output_group_size;
    Tensor& w_new = *ctx->Output(output_start_index, w.Shape());
    Tensor& m1_new = *ctx->Output(output_start_index + 1, w.Shape());
    Tensor& m2_new = *ctx->Output(output_start_index + 2, w.Shape());

    ORT_ENFORCE(w.Shape() == m1.Shape());
    ORT_ENFORCE(w.Shape() == g.Shape());
    ORT_ENFORCE(w.Shape() == m2.Shape());
    ORT_ENFORCE(w.Shape() == m1_new.Shape());
    ORT_ENFORCE(w.Shape() == m2_new.Shape());

    // We should throw for overflow in reduction APIs.
    // The index in CUDA system is integer.
    ORT_ENFORCE(
        w.Shape().Size() <
        static_cast<int64_t>(std::numeric_limits<int>::max()));

    const int size = static_cast<int>(w.Shape().Size());

    // Cast input tensors to pointers because CUDA kernels consume pointers.
    const CudaT2* w_data = reinterpret_cast<const CudaT2*>(w.template Data<T2>());
    const CudaT3* g_data = reinterpret_cast<const CudaT3*>(g.template Data<T3>());
    const CudaT4* m1_data = reinterpret_cast<const CudaT4*>(m1.template Data<T4>());
    const CudaT4* m2_data = reinterpret_cast<const CudaT4*>(m2.template Data<T4>());

    // Cast output tensors to pointers because CUDA kernels consume pointers.
    CudaT2* w_new_data = reinterpret_cast<CudaT2*>(w_new.template MutableData<T2>());
    CudaT4* m1_new_data = reinterpret_cast<CudaT4*>(m1_new.template MutableData<T4>());
    CudaT4* m2_new_data = reinterpret_cast<CudaT4*>(m2_new.template MutableData<T4>());

    // Special processing for float16 weight because it could be null. 
    half* w_fp16_new_data = nullptr;
    if (ctx->Output(output_start_index + 3, w.Shape())) {
      Tensor& w_fp16_new = *ctx->Output(output_start_index + 3, w.Shape());
      ORT_ENFORCE(w.Shape() == w_fp16_new.Shape());
      w_fp16_new_data = reinterpret_cast<half*>(w_fp16_new.template MutableData<MLFloat16>());
    }

    // Prepare temporal memory for storing update direction.
    IAllocatorUniquePtr<T3> d = GetScratchBuffer<T3>(size);
    CudaT3* d_data = reinterpret_cast<CudaT3*>(d.get());

    // Lamb Stage 1.
    LambComputeDirectionImpl(
        w_data,
        g_data,
        m1_data,
        m2_data,
        loss_scale_data,
        ToCudaType<T4>::FromFloat(alpha_[group_index]),
        ToCudaType<T4>::FromFloat(beta_[group_index]),
        ToCudaType<T2>::FromFloat(lambda_[group_index]),
        ToCudaType<T4>::FromFloat(epsilon_[group_index]),
        d_data,
        m1_new_data,
        m2_new_data,
        size);

    // Allocate buffer for reduction computation of update direction.
    // We reduce type T3 tensor to type T2 scalar. An example is that T3=float16
    // and T2=float.
    IAllocatorUniquePtr<T2> d_norm_buffer = GetScratchBuffer<T2>(1);
    CudaT2* d_norm_data = reinterpret_cast<CudaT2*>(d_norm_buffer.get());

    // Allocate buffer for reduction computation of weight tensor.
    // We reduce type T2 tensor to type T2 scalar. An example is that T2=float.
    IAllocatorUniquePtr<T2> w_norm_buffer = GetScratchBuffer<T2>(1);
    CudaT2* w_norm_data = reinterpret_cast<CudaT2*>(w_norm_buffer.get());

    // Compute buffer size in byte for reduction APIs.
    const auto buffer_size = static_cast<size_t>(
        compute_reduction_buffer_size(
            static_cast<int>(sizeof(T2)), size));

    // Allocate reduction buffer whose size is buffer_size bytes.
    IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(buffer_size);
    CudaT2* reduction_data = reinterpret_cast<CudaT2*>(reduction_buffer.get());

    // Lamb reduction Stage.
    reduce_l2_norm(
        w_data,
        w_norm_data,
        size,
        reduction_data);

    reduce_l2_norm(
        d_data,
        d_norm_data,
        size,
        reduction_data);


    // Lamb reduction Stage 2.
    // Use the update direction and the computed norms to compute
    // the new weights.
    LambUpdateImpl(
        eta_data,
        d_norm_data,
        w_norm_data,
        w_data,
        ToCudaType<T2>::FromFloat(threshold_[group_index]),
        d_data,
        w_new_data,
        w_fp16_new_data,
        size);
  }

  return Status::OK();
}

template <typename T, typename T_GRAD>
Status AccumulateGradient<T, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

  const Tensor& gradient_buffer = *ctx->Input<Tensor>(0);
  const Tensor& gradient = *ctx->Input<Tensor>(1);
  Tensor& accumulated_gradient = *ctx->Output(0, gradient_buffer.Shape());

  AccumulateGradientImpl(
      reinterpret_cast<const CudaT*>(gradient_buffer.template Data<T>()),
      reinterpret_cast<const CudaT_GRAD*>(gradient.template Data<T_GRAD>()),
      reinterpret_cast<CudaT*>(accumulated_gradient.template MutableData<T>()),
      gradient.Shape().Size());

  return Status::OK();
}

#define REGISTER_GRADIENT_ACCUMULATOR_TYPED(T, T_GRAD)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      GradientAccumulator,                                                  \
      kOnnxDomain,                                                          \
      9,                                                                    \
      T##_##T_GRAD,                                                         \
      kCudaExecutionProvider,                                               \
      KernelDefBuilder()                                                    \
          .Alias(0, 0) /* Accumulate gradients in-place */                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()), \
      AccumulateGradient<T, T_GRAD>);
REGISTER_GRADIENT_ACCUMULATOR_TYPED(float, float)
REGISTER_GRADIENT_ACCUMULATOR_TYPED(float, MLFloat16)

template <typename T>
Status ZeroGradient<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& old_gradient = *ctx->Input<Tensor>(0);
  Tensor& zero_gradient = *ctx->Output(0, old_gradient.Shape());

  CUDA_RETURN_IF_ERROR(cudaMemset(zero_gradient.template MutableData<T>(), 0, zero_gradient.Shape().Size() * sizeof(T)));
  return Status::OK();
}

#define REGISTER_ZERO_GRADIENT_TYPED(T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ZeroGradient,                                               \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .Alias(0, 0) /* Zero out gradients in-place */          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", DataTypeImpl::AllTensorTypes()),  \
      ZeroGradient<T>);
REGISTER_ZERO_GRADIENT_TYPED(float)
REGISTER_ZERO_GRADIENT_TYPED(MLFloat16)

}  // namespace cuda
}  // namespace onnxruntime
