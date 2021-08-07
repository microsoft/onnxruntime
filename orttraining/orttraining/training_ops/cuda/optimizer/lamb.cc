// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "orttraining/training_ops/cuda/optimizer/common.h"
#include "orttraining/training_ops/cuda/optimizer/lamb.h"

namespace onnxruntime {
namespace cuda {

std::vector<std::pair<int, int>> GenerateLambExtraAliasMapping() {
  // Starting index of extra inputs.
  constexpr int input_index_bias = 5;
  // Starting index of extra outputs.
  constexpr int output_index_bias = 1;
  // Count of extra I/O groups. One group corresponds to a weight update.
  constexpr int group_count = 1024;
  // length of [w, g, m1, m2, w_mixed_precision].
  constexpr int input_stride = 5;
  // length of [w_new, g_new, m1_new, m2_new, w_mixed_precision_new].
  constexpr int output_stride = 5;

  std::vector<std::pair<int, int>> alias_pairs{};
  for (int i = 0; i < group_count; ++i) {
    const int input = input_index_bias + i * input_stride;
    const int output = output_index_bias + i * output_stride;
    // w --> w_new
    alias_pairs.emplace_back(std::make_pair(input, output));
    // g --> g_new
    alias_pairs.emplace_back(std::make_pair(input + 1, output + 1));
    // m1 --> m1_new
    alias_pairs.emplace_back(std::make_pair(input + 2, output + 2));
    // m2 --> m2_new
    alias_pairs.emplace_back(std::make_pair(input + 3, output + 3));
    // w_mixed_precision --> w_mixed_precision_new
    alias_pairs.emplace_back(std::make_pair(input + 4, output + 4));
  }

  // update_count are updated in place.
  alias_pairs.emplace_back(std::make_pair(4, 0));

  return alias_pairs;
}

// TODO: Once Schema is checked in to onnx lets fix this to match that
#define REGISTER_LAMB_KERNEL_TYPED(T1, T2, T3, T4, T_GRAD_NORM, T_MIXED_PRECISION_FP)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                       \
      LambOptimizer,                                                                                   \
      kMSDomain,                                                                                       \
      1,                                                                                               \
      T1##_##T2##_##T3##_##T4##_##T_GRAD_NORM##_##T_MIXED_PRECISION_FP,                                \
      kCudaExecutionProvider,                                                                          \
      (*KernelDefBuilder::Create())                                                                    \
          .Alias(GenerateLambExtraAliasMapping())                                                      \
          .InputMemoryType(OrtMemTypeCPUInput, 0)   /* Keep do_update in CPU */                        \
          .InputMemoryType(OrtMemTypeCPUInput, 4)   /* Keep iteration_count in CPU */                  \
          .OutputMemoryType(OrtMemTypeCPUOutput, 0) /* Keep iteration_count in CPU */                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                                     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                                     \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())                                     \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())                                     \
          .TypeConstraint("T_MIXED_PRECISION_FP", DataTypeImpl::GetTensorType<T_MIXED_PRECISION_FP>()) \
          .TypeConstraint("T_GRAD_NORM", DataTypeImpl::GetTensorType<T_GRAD_NORM>()),                  \
      LambOptimizer<T1, T2, T3, T4, T_GRAD_NORM, T_MIXED_PRECISION_FP>);

REGISTER_LAMB_KERNEL_TYPED(float, float, MLFloat16, float, MLFloat16, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(float, float, MLFloat16, float, float, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(float, float, float, float, float, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(double, double, double, double, double, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, MLFloat16, MLFloat16, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, MLFloat16, float, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, float, MLFloat16, MLFloat16)
REGISTER_LAMB_KERNEL_TYPED(MLFloat16, float, MLFloat16, float, float, MLFloat16)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_LAMB_KERNEL_TYPED(float, float, BFloat16, float, BFloat16, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(float, float, BFloat16, float, float, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(float, float, float, float, float, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(double, double, double, double, double, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(BFloat16, float, BFloat16, BFloat16, BFloat16, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(BFloat16, float, BFloat16, BFloat16, float, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(BFloat16, float, BFloat16, float, BFloat16, BFloat16)
REGISTER_LAMB_KERNEL_TYPED(BFloat16, float, BFloat16, float, float, BFloat16)
#endif

void check_inputs_and_outputs(
    const Tensor* w,
    const Tensor* g,
    const Tensor* m1,
    const Tensor* m2,
    const Tensor* w_mixed_precision,
    const Tensor* w_new,
    const Tensor* g_new,
    const Tensor* m1_new,
    const Tensor* m2_new,
    const Tensor* w_mixed_precision_new) {
  // Throw if we have incomplete input or output lists.
  ORT_ENFORCE(w, "Weight tensor should not be null.");
  ORT_ENFORCE(g, "gradient tensor should not be null.");
  ORT_ENFORCE(m1, "First-order momentum tensor should not be null.");
  ORT_ENFORCE(m2, "Second-order momentum tensor should not be null.");
  ORT_ENFORCE(m1_new, "New first-order momentum tensor should not be null.");
  ORT_ENFORCE(m2_new, "New second-order momentum tensor should not be null.");
  // Check if all shapes are good.
  ORT_ENFORCE(m1->Shape() == m1_new->Shape());
  ORT_ENFORCE(m2->Shape() == m2_new->Shape());
  if (w_new)
    ORT_ENFORCE(w->Shape() == w_new->Shape());
  if (g_new)
    ORT_ENFORCE(g->Shape() == g_new->Shape());
  if (w_mixed_precision && w_mixed_precision_new)
    ORT_ENFORCE(w_mixed_precision->Shape() == w_mixed_precision_new->Shape());
}

template <typename TWeight, typename TGradient, typename TMomentum, typename TMixedPrecision>
Status copy_inputs_to_outputs(
    cudaStream_t stream,
    OpKernelContext* ctx,
    const int non_grouped_input_count,
    const int non_grouped_output_count,
    const int group_count,
    const int input_group_size,
    const int output_group_size) {
  const Tensor* step_tensor = ctx->Input<Tensor>(4);
  if (step_tensor) {
    const int64_t* step_data = step_tensor->template Data<int64_t>();
    Tensor* step_tensor_new = ctx->Output(0, step_tensor->Shape());
    ORT_ENFORCE(step_tensor_new != nullptr, "Step tensor (input) and updated step tensor (output) must be specified together.");
    int64_t* step_data_new = step_tensor_new->template MutableData<int64_t>();
    *step_data_new = *step_data;
  }

  for (int group_index = 0; group_index < group_count; ++group_index) {
    const int input_start_index = non_grouped_input_count + group_index * input_group_size;
    const Tensor& w = *ctx->Input<Tensor>(input_start_index);
    const Tensor& g = *ctx->Input<Tensor>(input_start_index + 1);
    const Tensor& m1 = *ctx->Input<Tensor>(input_start_index + 2);
    const Tensor& m2 = *ctx->Input<Tensor>(input_start_index + 3);
    const Tensor* w_mixed_precision = ctx->Input<Tensor>(input_start_index + 4);
    const int output_start_index = non_grouped_output_count + group_index * output_group_size;
    Tensor* w_new = ctx->Output(output_start_index, w.Shape());
    Tensor* g_new = ctx->Output(output_start_index + 1, g.Shape());
    Tensor& m1_new = *ctx->Output(output_start_index + 2, m1.Shape());
    Tensor& m2_new = *ctx->Output(output_start_index + 3, m2.Shape());
    Tensor* w_mixed_precision_new = w_mixed_precision != nullptr ? ctx->Output(output_start_index + 4, w_mixed_precision->Shape()) : nullptr;

    // TODO: temporary hack until View is improved (it doesn't work with Alias)
    if (w_new != nullptr)
      w_new->SetByteOffset(w.ByteOffset());
    if (g_new != nullptr)
      g_new->SetByteOffset(g.ByteOffset());
    if (w_mixed_precision_new != nullptr)
      w_mixed_precision_new->SetByteOffset(w_mixed_precision->ByteOffset());

    if (w_new) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<TWeight>(stream, w, *w_new));
    }
    if (g_new) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<TGradient>(stream, g, *g_new));
    }
    ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<TMomentum>(stream, m1, m1_new));
    ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<TMomentum>(stream, m2, m2_new));

    if (w_mixed_precision_new) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<TMixedPrecision>(stream, *w_mixed_precision, *w_mixed_precision_new));
    }
  }

  return Status::OK();
}

template <typename CudaT2, typename CudaT3, typename CudaT4, typename CudaT_GRAD_NORM>
Status launch_lamb_compute_direction(
    cudaStream_t stream,
    const int64_t update_count,
    const int group_count,
    const CudaT2* p_loss_scale,
    const CudaT_GRAD_NORM* p_g_norm,
    std::vector<int>& tensor_sizes,
    std::vector<const CudaT2*>& p_ws,
    std::vector<const CudaT3*>& p_gs,
    std::vector<const CudaT4*>& p_m1s,
    std::vector<const CudaT4*>& p_m2s,
    std::vector<CudaT3*>& p_ds,
    std::vector<CudaT4*>& p_m1_news,
    std::vector<CudaT4*>& p_m2_news,
    const std::vector<float>& alphas,
    const std::vector<float>& betas,
    const std::vector<float>& lambdas,
    const std::vector<float>& epsilons,
    const std::vector<float>& max_norms,
    const int64_t do_bias_correction) {
  ORT_ENFORCE(group_count == static_cast<int>(tensor_sizes.size()));

  ORT_ENFORCE(group_count == static_cast<int>(p_ws.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_gs.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_m1s.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_m2s.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_ds.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_m1_news.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_m2_news.size()));

  ORT_ENFORCE(group_count == static_cast<int>(alphas.size()));
  ORT_ENFORCE(group_count == static_cast<int>(betas.size()));
  ORT_ENFORCE(group_count == static_cast<int>(lambdas.size()));
  ORT_ENFORCE(group_count == static_cast<int>(epsilons.size()));

  constexpr int tensor_count_per_group = 6;
  const int max_tensor_size = compute_max_tensor_size_per_launch<tensor_count_per_group>(4);
  // Bucketize tensor groups by the associated optimizer configuration.
  // If two tensor groups use different "alpha", they should be put into two distinct buckets.
  std::map<std::tuple<float, float, float, float, float>, std::vector<std::vector<void*>>> buckets;
  std::map<std::tuple<float, float, float, float, float>, std::vector<int>> tensor_sizes_in_buckets;
  for (int i = 0; i < group_count; ++i) {
    if (tensor_sizes[i] > max_tensor_size) {
      // For the first iteration (indexed by 0), the update count should be 2.
      const float alpha_correction =
          do_bias_correction ? onnxruntime::contrib::compute_bias_correction_coefficient(alphas[i], update_count) : 1.f;
      const float beta_correction =
          do_bias_correction ? onnxruntime::contrib::compute_bias_correction_coefficient(betas[i], update_count) : 1.f;

      LambComputeDirection(
          stream,
          p_ws[i],
          p_gs[i],
          p_m1s[i],
          p_m2s[i],
          p_loss_scale,
          p_g_norm,
          alphas[i],
          betas[i],
          lambdas[i],
          epsilons[i],
          max_norms[i],
          alpha_correction,
          beta_correction,
          p_ds[i],
          p_m1_news[i],
          p_m2_news[i],
          tensor_sizes[i]);
    } else {
      std::vector<void*> ptrs(tensor_count_per_group);
      ptrs[0] = const_cast<CudaT2*>(p_ws[i]);   // weight tensor
      ptrs[1] = const_cast<CudaT3*>(p_gs[i]);   // gradient (reused to store update direction)
      ptrs[2] = const_cast<CudaT4*>(p_m1s[i]);  // 1st momentum
      ptrs[3] = const_cast<CudaT4*>(p_m2s[i]);  // 2nd momentum
      ptrs[4] = p_m1_news[i];                   // new 1st momentum
      ptrs[5] = p_m2_news[i];                   // new 2nd momentum

      auto key = std::make_tuple(alphas[i], betas[i], lambdas[i], epsilons[i], max_norms[i]);
      buckets[key].push_back(ptrs);
      tensor_sizes_in_buckets[key].push_back(tensor_sizes[i]);
    }
  }

  for (auto& pair : buckets) {
    const auto key = pair.first;
    float alpha = 0.f, beta = 0.f, lambda = 0.f, epsilon = 0.f, max_norm = 0.f;
    std::tie(alpha, beta, lambda, epsilon, max_norm) = key;

    // For the first iteration (indexed by 0), the update count should be 1.
    const float alpha_correction =
        do_bias_correction ? onnxruntime::contrib::compute_bias_correction_coefficient(alpha, update_count) : 1.f;
    const float beta_correction =
        do_bias_correction ? onnxruntime::contrib::compute_bias_correction_coefficient(beta, update_count) : 1.f;

    typedef LambMultiTensorComputeDirectionFunctor<CudaT2, CudaT3, CudaT4, CudaT_GRAD_NORM> LambStage1;
    LambStage1 lamb_stage1;

    launch_multi_tensor_functor<tensor_count_per_group, LambStage1>(
        stream,
        2048 * 32,
        tensor_sizes_in_buckets[key],
        buckets[key],
        lamb_stage1,
        p_loss_scale, p_g_norm, lambda, alpha, beta, epsilon, max_norm, alpha_correction, beta_correction);
  }

  return Status::OK();
}

template <typename CudaTNorm, typename CudaTIn1, typename CudaTIn2>
Status launch_lamb_reduction(
    const CudaKernel& kernel,
    const int group_count,
    std::vector<int>& tensor_sizes,
    std::vector<CudaTNorm*>& p_w_norms,
    std::vector<CudaTNorm*>& p_d_norms,
    std::vector<const CudaTIn1*>& p_ws,
    std::vector<CudaTIn2*>& p_ds,
    void* reduction_buffer,
    size_t reduction_buffer_size) {
  ORT_ENFORCE(group_count == static_cast<int>(tensor_sizes.size()));

  ORT_ENFORCE(group_count == static_cast<int>(p_w_norms.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_d_norms.size()));

  ORT_ENFORCE(group_count == static_cast<int>(p_ws.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_ds.size()));

  constexpr int tensor_count_per_group = 4;

  cudaStream_t stream = kernel.Stream();
  // Bucketize tensor groups by the associated optimizer configuration.
  // If two tensor groups use different "alpha", they should be put into two distinct buckets.
  std::vector<std::vector<void*>> buckets;
  std::vector<int> tensor_sizes_in_buckets;
  const int max_tensor_size = compute_max_tensor_size_per_launch<tensor_count_per_group>(4);
  for (int i = 0; i < group_count; ++i) {
    if (tensor_sizes[i] > max_tensor_size) {
      ORT_RETURN_IF_ERROR(reduce_square_sum(
          stream,
          p_ws[i],
          p_w_norms[i],
          tensor_sizes[i],
          reduction_buffer,
          reduction_buffer_size));
      ORT_RETURN_IF_ERROR(reduce_square_sum(
          stream,
          p_ds[i],
          p_d_norms[i],
          tensor_sizes[i],
          reduction_buffer,
          reduction_buffer_size));
    } else {
      std::vector<void*> ptrs(tensor_count_per_group);
      ptrs[0] = const_cast<CudaTIn1*>(p_ws[i]);  // weight tensor
      ptrs[1] = const_cast<CudaTIn2*>(p_ds[i]);  // update direction
      ptrs[2] = p_w_norms[i];                    // weight tensor's norm
      ptrs[3] = p_d_norms[i];                    // update direction's norm

      buckets.push_back(ptrs);
      tensor_sizes_in_buckets.push_back(tensor_sizes[i]);
    }
  }

  if (buckets.size() > 0) {
    ORT_ENFORCE(tensor_sizes_in_buckets.size() > 0);
  }

  if (tensor_sizes_in_buckets.size() > 0) {
    ORT_ENFORCE(buckets.size() > 0);
  }

  // Only launch multi-tensor function if we have at least one tensor in the buckets.
  if (tensor_sizes_in_buckets.size() > 0 && buckets.size() > 0) {
    typedef LambMultiTensorReductionFunctor<CudaTIn1, CudaTIn2, CudaTNorm, CudaTNorm, CudaTNorm> TReducer;
    TReducer reducer;
    launch_multi_tensor_functor<tensor_count_per_group, TReducer>(
        stream,
        2048 * 32,
        tensor_sizes_in_buckets,
        buckets,
        reducer,
        kernel,
        reduction_buffer,
        reduction_buffer_size);
  }

  return Status::OK();
}

template <typename CudaT1, typename CudaT2, typename CudaT3, typename CudaT_MIXED_PRECISION_FP>
Status launch_lamb_update(
    cudaStream_t stream,
    const int group_count,
    const CudaT1* eta,
    const float ratio_min,
    const float ratio_max,
    std::vector<int>& tensor_sizes,
    std::vector<CudaT2*>& p_w_norms,
    std::vector<CudaT2*>& p_d_norms,
    std::vector<const CudaT2*>& p_ws,
    std::vector<CudaT3*>& p_ds,
    /* output */ std::vector<CudaT2*>& p_w_news,
    /* output */ std::vector<CudaT3*>& p_g_news,
    /* output */ std::vector<CudaT_MIXED_PRECISION_FP*>& p_w_mixed_precision_news) {
  ORT_ENFORCE(group_count == static_cast<int>(tensor_sizes.size()));

  ORT_ENFORCE(group_count == static_cast<int>(p_w_norms.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_d_norms.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_ws.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_ds.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_w_news.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_g_news.size()));
  ORT_ENFORCE(group_count == static_cast<int>(p_w_mixed_precision_news.size()));

  constexpr int tensor_count_per_group = 7;

  // Bucketize tensor groups by the associated optimizer configuration.
  // If two tensor groups use different "alpha", they should be put into two distinct buckets.
  std::vector<std::vector<void*>> buckets;
  std::vector<int> tensor_sizes_in_bucket;
  const int max_tensor_size = compute_max_tensor_size_per_launch<tensor_count_per_group>(4);
  for (int i = 0; i < group_count; ++i) {
    if (tensor_sizes[i] > max_tensor_size) {
      LambUpdate(
          stream,
          eta,
          ratio_min,
          ratio_max,
          p_d_norms[i],
          p_w_norms[i],
          p_ws[i],
          p_ds[i],
          p_w_news[i],
          p_g_news[i],
          p_w_mixed_precision_news[i],
          tensor_sizes[i]);
    } else {
      std::vector<void*> ptrs(tensor_count_per_group);
      ptrs[0] = p_w_norms[i];                  // weight tensor's norm
      ptrs[1] = p_d_norms[i];                  // direction's norm
      ptrs[2] = const_cast<CudaT2*>(p_ws[i]);  // weight tensor
      ptrs[3] = p_ds[i];                       // direction
      ptrs[4] = p_w_news[i];                   // new weight tensor
      ptrs[5] = p_g_news[i];                   // new gradient tensor
      ptrs[6] = p_w_mixed_precision_news[i];   // new half-precision weight tensor
      buckets.push_back(ptrs);
      tensor_sizes_in_bucket.push_back(tensor_sizes[i]);
    }
  }

  if (buckets.size() > 0) {
    ORT_ENFORCE(tensor_sizes_in_bucket.size() > 0);
  }

  if (tensor_sizes_in_bucket.size() > 0) {
    ORT_ENFORCE(buckets.size() > 0);
  }

  // Only launch multi-tensor function if we have at least one tensor in the buckets.
  if (tensor_sizes_in_bucket.size() > 0 && buckets.size() > 0) {
    typedef LambMultiTensorUpdateFunctor<
        CudaT1, CudaT2, CudaT3, CudaT_MIXED_PRECISION_FP>
        LambStage2;
    LambStage2 lamb_stage2;

    launch_multi_tensor_functor<tensor_count_per_group, LambStage2>(
        stream,
        2048 * 32,
        tensor_sizes_in_bucket,
        buckets,
        lamb_stage2,
        eta,
        ratio_min,
        ratio_max);
  }

  return Status::OK();
}

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD_NORM, typename T_MIXED_PRECISION_FP>
Status LambOptimizer<T1, T2, T3, T4, T_GRAD_NORM, T_MIXED_PRECISION_FP>::ComputeInternal(OpKernelContext* ctx) const {
  // CudaT* are types used to invoke CUDA-based functions. It, for example, maps
  // MLFloat16 in ONNXRuntime to half in CUDA.
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;
  typedef typename ToCudaType<T3>::MappedType CudaT3;
  typedef typename ToCudaType<T4>::MappedType CudaT4;
  typedef typename ToCudaType<T_GRAD_NORM>::MappedType CudaT_GRAD_NORM;
  typedef typename ToCudaType<T_MIXED_PRECISION_FP>::MappedType CudaT_MIXED_PRECISION_FP;

  constexpr int non_grouped_input_count = 5;
  constexpr int input_group_size = 5;
  constexpr int output_group_size = 5;
  constexpr int non_grouped_output_count = 1;
  constexpr int minimal_input_count = non_grouped_input_count + 1 * input_group_size - 1;
  constexpr int minimal_output_count = non_grouped_output_count + 1 * output_group_size - 1;
  const int grouped_input_tensor_count = ctx->InputCount() - non_grouped_input_count;
  const int grouped_output_tensor_count = ctx->OutputCount() - non_grouped_output_count;

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

  // In addition to the first non_grouped_input_count inputs, all inputs are repeated sequence of [w, g, m1, m2, w_mixed_precision].
  ORT_ENFORCE(
      grouped_input_tensor_count % input_group_size == 0,
      "Input count must be ", non_grouped_input_count, " + ", input_group_size,
      " x (number of weights to optimize).");
  // Outputs are repeated sequence of [w_new, g_new, m1_new, m2_new, w_mixed_precision_new].
  ORT_ENFORCE(
      grouped_output_tensor_count % output_group_size == 0,
      "Output count must be ", non_grouped_output_count, " + ", output_group_size,
      " x (number of weights to optimize).");
  // Number of repeated [w, g, m1, m2, w_mixed_precision]'s should match number of repeated [w_new, g_new, m1_new, m2_new, w_mixed_precision_new].
  ORT_ENFORCE(
      grouped_input_tensor_count / input_group_size == grouped_output_tensor_count / output_group_size,
      "Input and output tensor counts are not aligned. Please check LambOptimizer's input and output lists.");

  // Number of [w, g, m1, m2, (w_mixed_precision)] (or [w_new, m1_new, m2_new, (w_mixed_precision_new)]).
  const int group_count = (grouped_input_tensor_count + input_group_size - 1) / input_group_size;

  // At least we need one group of alpha, beta, lambda, ..., for processing one group.
  ORT_ENFORCE(alpha_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(beta_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(lambda_.size() >= static_cast<size_t>(group_count));
  ORT_ENFORCE(epsilon_.size() >= static_cast<size_t>(group_count));

  // If gradient norm is not finite, we copy inputs to outputs directly.
  if (ctx->Input<Tensor>(0)) {
    auto update_signal_tensor = ctx->Input<Tensor>(0);
    auto update_signal = *update_signal_tensor->template Data<bool>();
    if (!update_signal) {
      return copy_inputs_to_outputs<T2, T3, T4, T_MIXED_PRECISION_FP>(
          Stream(),
          ctx,
          non_grouped_input_count,
          non_grouped_output_count,
          group_count,
          input_group_size,
          output_group_size);
    }
  }

  const CudaT2* loss_scale_data = nullptr;
  if (ctx->Input<Tensor>(1)) {
    const Tensor& loss_scale_tensor = *ctx->Input<Tensor>(1);
    loss_scale_data = reinterpret_cast<const CudaT2*>(loss_scale_tensor.template Data<T2>());
  }

  const CudaT_GRAD_NORM* g_norm_data = nullptr;
  if (ctx->Input<Tensor>(2)) {
    const Tensor& g_norm_tensor = *ctx->Input<Tensor>(2);
    g_norm_data = reinterpret_cast<const CudaT_GRAD_NORM*>(g_norm_tensor.template Data<T_GRAD_NORM>());
  }

  const Tensor& eta = *ctx->Input<Tensor>(3);
  const CudaT1* eta_data = reinterpret_cast<const CudaT1*>(eta.template Data<T1>());

  const Tensor* step_tensor = ctx->Input<Tensor>(4);
  const int64_t* step_data = nullptr;
  if (step_tensor) {
    step_data = step_tensor->template Data<int64_t>();
  }

  // Allocate buffer for reduction computation of update directions.
  // The i-th update direction's norm is stored at the i-th element.
  // We reduce type T3 tensor to type T2 scalar. An example is that T3=float16
  // and T2=float.
  IAllocatorUniquePtr<T2> d_norm_buffer = GetScratchBuffer<T2>(group_count);
  CudaT2* d_norm_data = reinterpret_cast<CudaT2*>(d_norm_buffer.get());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(d_norm_data, 0, group_count * sizeof(T2), Stream()));

  // Allocate buffer for reduction computation of weight tensor.
  // The i-th weight's norm is stored at the i-th element.
  // We reduce type T2 tensor to type T2 scalar. An example is that T2=float.
  IAllocatorUniquePtr<T2> w_norm_buffer = GetScratchBuffer<T2>(group_count);
  CudaT2* w_norm_data = reinterpret_cast<CudaT2*>(w_norm_buffer.get());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(w_norm_data, 0, group_count * sizeof(T2), Stream()));

  // Find the max size of updated weight tensors.
  int max_tensor_size = 0;
  for (int group_index = 0; group_index < group_count; ++group_index) {
    // Prepare used input tensors for this group.
    const int input_start_index = non_grouped_input_count + group_index * input_group_size;
    const Tensor& w = *ctx->Input<Tensor>(input_start_index);
    max_tensor_size = std::max(max_tensor_size, static_cast<int>(w.Shape().Size()));
  }

  const size_t reduction_buffer_size = [&]() {
    // Allocate a buffer in byte for reduction API calls.
    size_t rbs = compute_reduction_buffer_size<CudaT2>(max_tensor_size);

    // Enlarge reduction buffer to accomodate multi-tensor reduction kernel as well
    const int tensor_group_size = 4;  // w, d, w_norm, d_norm
    const int max_blocks = ChunkGroup<tensor_group_size>::max_block_count;
    const size_t multitensor_block_reduce_buffer_size = 2 * max_blocks * sizeof(CudaT2);
    rbs = std::max(rbs, multitensor_block_reduce_buffer_size);

    return rbs;
  }();

  // Allocate reduction buffer whose size is reduction_buffer_size bytes.
  IAllocatorUniquePtr<void> reduction_buffer = GetScratchBuffer<void>(reduction_buffer_size);

  // Input tensors' pointers.
  std::vector<const CudaT2*> p_ws(group_count);
  std::vector<const CudaT3*> p_gs(group_count);
  std::vector<const CudaT4*> p_m1s(group_count);
  std::vector<const CudaT4*> p_m2s(group_count);
  std::vector<const CudaT_MIXED_PRECISION_FP*> p_w_mixed_precisions(group_count);
  // ds' is an mutable version of gs' because we want to reuse
  // gs' memory to store the update direction to avoid allocating a model-scale buffer.
  std::vector<CudaT3*> p_ds(group_count);
  // Intermediate tensors, weight tensors' and directions' norms.
  std::vector<CudaT2*> p_w_norms(group_count);
  std::vector<CudaT2*> p_d_norms(group_count);
  // Output tensors' pointers.
  std::vector<CudaT2*> p_w_news(group_count);
  std::vector<CudaT3*> p_g_news(group_count);
  std::vector<CudaT4*> p_m1_news(group_count);
  std::vector<CudaT4*> p_m2_news(group_count);
  std::vector<CudaT_MIXED_PRECISION_FP*> p_w_mixed_precision_news(group_count);
  // The i-th element in following array is the size of
  // the i-th updated weight tensor and other related tensors.
  std::vector<int> tensor_sizes(group_count);

  for (int group_index = 0; group_index < group_count; ++group_index) {
    // Prepare used input tensors for this group.
    const int input_start_index = non_grouped_input_count + group_index * input_group_size;
    const Tensor* w = ctx->Input<Tensor>(input_start_index);
    const Tensor* g = ctx->Input<Tensor>(input_start_index + 1);
    const Tensor* m1 = ctx->Input<Tensor>(input_start_index + 2);
    const Tensor* m2 = ctx->Input<Tensor>(input_start_index + 3);
    const Tensor* w_mixed_precision = ctx->Input<Tensor>(input_start_index + 4);

    // Prepare used outputs tensors for this group.
    const int output_start_index = non_grouped_output_count + group_index * output_group_size;
    Tensor* w_new = ctx->Output(output_start_index, w->Shape());
    Tensor* g_new = ctx->Output(output_start_index + 1, g->Shape());
    Tensor* m1_new = ctx->Output(output_start_index + 2, m1->Shape());
    Tensor* m2_new = ctx->Output(output_start_index + 3, m2->Shape());
    Tensor* w_mixed_precision_new = w_mixed_precision != nullptr ? ctx->Output(output_start_index + 4, w_mixed_precision->Shape()) : nullptr;

    // TODO: temporary hack until View is improved (it doesn't work with Alias)
    if (w_new != nullptr)
      w_new->SetByteOffset(w->ByteOffset());
    if (g_new != nullptr)
      g_new->SetByteOffset(g->ByteOffset());
    if (w_mixed_precision_new != nullptr)
      w_mixed_precision_new->SetByteOffset(w_mixed_precision->ByteOffset());

    check_inputs_and_outputs(w, g, m1, m2, w_mixed_precision, w_new, g_new, m1_new, m2_new, w_mixed_precision_new);

    // We should throw for preventing overflow in reduction APIs.
    // The index in CUDA system is 32-bit integer.
    ORT_ENFORCE(
        w->Shape().Size() <
        static_cast<int64_t>(std::numeric_limits<int>::max()));
    tensor_sizes[group_index] = static_cast<int>(w->Shape().Size());

    // Input tensors' pointers.
    p_ws[group_index] = reinterpret_cast<const CudaT2*>(w->template Data<T2>());
    p_gs[group_index] = reinterpret_cast<const CudaT3*>(g->template Data<T3>());
    p_m1s[group_index] = reinterpret_cast<const CudaT4*>(m1->template Data<T4>());
    p_m2s[group_index] = reinterpret_cast<const CudaT4*>(m2->template Data<T4>());
    p_w_mixed_precisions[group_index] = w_mixed_precision != nullptr ? reinterpret_cast<const CudaT_MIXED_PRECISION_FP*>(w_mixed_precision->template Data<T_MIXED_PRECISION_FP>()) : nullptr;

    // The following cast is for reusing gradient tensor g to store update direction d.
    p_ds[group_index] = const_cast<CudaT3*>(reinterpret_cast<const CudaT3*>(g->template Data<T3>()));

    // Set up which pointer to store which tensor's norm.
    p_w_norms[group_index] = w_norm_data + group_index;
    p_d_norms[group_index] = d_norm_data + group_index;

    // Output tensors' pointers.
    p_w_news[group_index] = w_new != nullptr ? reinterpret_cast<CudaT2*>(w_new->template MutableData<T2>()) : nullptr;
    p_g_news[group_index] = g_new != nullptr ? reinterpret_cast<CudaT3*>(g_new->template MutableData<T3>()) : nullptr;
    p_m1_news[group_index] = reinterpret_cast<CudaT4*>(m1_new->template MutableData<T4>());
    p_m2_news[group_index] = reinterpret_cast<CudaT4*>(m2_new->template MutableData<T4>());
    p_w_mixed_precision_news[group_index] = w_mixed_precision_new != nullptr ? reinterpret_cast<CudaT_MIXED_PRECISION_FP*>(w_mixed_precision_new->template MutableData<T_MIXED_PRECISION_FP>()) : nullptr;
  }

  ORT_RETURN_IF_ERROR(launch_lamb_compute_direction(
      Stream(),
      step_data ? *step_data : 0,
      group_count,
      loss_scale_data,
      g_norm_data,
      tensor_sizes,
      p_ws, p_gs, p_m1s, p_m2s,
      p_ds,
      p_m1_news, p_m2_news,
      alpha_, beta_, lambda_, epsilon_, max_norm_clip_,
      do_bias_correction_));

  ORT_RETURN_IF_ERROR(launch_lamb_reduction(
      *this,
      group_count,
      tensor_sizes,
      p_w_norms,
      p_d_norms,
      p_ws,
      p_ds,
      reduction_buffer.get(),
      reduction_buffer_size));

  ORT_RETURN_IF_ERROR(launch_lamb_update(
      Stream(),
      group_count,
      eta_data,
      ratio_min_,
      ratio_max_,
      tensor_sizes,
      p_w_norms,
      p_d_norms,
      p_ws,
      p_ds,
      p_w_news,
      p_g_news,
      p_w_mixed_precision_news));

  if (step_tensor) {
    Tensor* step_tensor_new = ctx->Output(0, step_tensor->Shape());
    ORT_ENFORCE(step_tensor_new != nullptr, "Step tensor (input) and updated step tensor (output) must be specified together.");
    int64_t* step_data_new = step_tensor_new->template MutableData<int64_t>();
    *step_data_new = *step_data + 1;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
