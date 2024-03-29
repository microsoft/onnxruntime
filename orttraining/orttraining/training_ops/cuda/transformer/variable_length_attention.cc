// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/transformer/variable_length_attention.h"

#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/nn/dropout_impl.h"
#ifndef SHARED_PROVIDER
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/common/gsl.h"
#endif
#include "core/framework/ort_value.h"
#include "orttraining/training_ops/cuda/transformer/variable_length_attention_impl.h"

#include "core/providers/cuda/math/softmax.h"

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "orttraining/training_ops/cuda/math/group_gemm_impl.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "core/providers/cuda/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      MultiHeadAttentionVarLength,                                          \
      kMSDomain,                                                            \
      1,                                                                    \
      T,                                                                    \
      kCudaExecutionProvider,                                               \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())      \
          .TypeConstraint("TI", std::vector<MLDataType>{                    \
                                    DataTypeImpl::GetTensorType<int64_t>(), \
                                })                                          \
          .InputMemoryType(OrtMemTypeCPUInput, 6),                          \
      MultiHeadAttentionVarLength<T>);

namespace {

constexpr static int seq_axis_on_input = 1;
constexpr static int transpose_input_dim_size = 4;

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DropoutComputeImpl {
  void operator()(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const int64_t mask_element_count,
                  const float ratio_data, PhiloxGenerator& generator, const Tensor& X, Tensor& Y, void* mask_data,
                  bool use_bitmask) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.Data<T>());
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.MutableData<T>());

    DropoutKernelImpl<CudaT>(prop, stream, N, mask_element_count, ratio_data, generator, X_data, Y_data, mask_data,
                             use_bitmask);
  }
};

}  // namespace

namespace {

OrtValue AllocateTensorInMLValue(const MLDataType data_type, const TensorShape& shape, AllocatorPtr& allocator) {
  auto new_tensor = Tensor::Create(data_type, shape, allocator);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  return OrtValue{new_tensor.release(), ml_tensor,
                  ml_tensor->GetDeleteFunc()};
}

void PrepareTransposeLaunchInfos(int variant_axis_on_input,
                                 const TArray<int>& perms,
                                 const TArray<int64_t>& full_sized_shape,
                                 TArray<int64_t>& full_sized_transposed_output_shape,
                                 int64_t& factor_for_fixed_dims,
                                 int& variant_axis_on_output) {
  factor_for_fixed_dims = 1;
  full_sized_transposed_output_shape.SetSize(full_sized_shape.Size());
  for (int32_t i = 0; i < full_sized_shape.Size(); ++i) {
    full_sized_transposed_output_shape[i] = full_sized_shape[perms[i]];
    if (i != variant_axis_on_input) {
      factor_for_fixed_dims *= full_sized_shape[i];
    }
  }

  variant_axis_on_output = perms[variant_axis_on_input];
}

}  // namespace

template <typename T>
Status MultiHeadAttentionVarLength<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* q_tensor = context->Input<Tensor>(0);                  // Shape: [token_count, head_count, hidden_size_per_head]
  const Tensor* k_tensor = context->Input<Tensor>(1);                  // Shape: [token_count, head_count, hidden_size_per_head]
  const Tensor* v_tensor = context->Input<Tensor>(2);                  // Shape: [token_count, head_count, hidden_size_per_head]
  const Tensor* q_cum_seqlen_tensor = context->Input<Tensor>(3);       // Shape: [batch_size + 1]
  const Tensor* k_cum_seqlen_tensor = context->Input<Tensor>(4);       // Shape: [batch_size + 1]
  const Tensor* v_cum_seqlen_tensor = context->Input<Tensor>(5);       // Shape: [batch_size + 1]
  const Tensor* bsz_and_seq_shape_tensor = context->Input<Tensor>(6);  // Used to unflatten the token_count.

  ORT_RETURN_IF(q_tensor == nullptr, "q_tensor != nullptr");
  ORT_RETURN_IF(k_tensor == nullptr, "k_tensor != nullptr");
  ORT_RETURN_IF(v_tensor == nullptr, "v_tensor != nullptr");
  ORT_RETURN_IF(q_cum_seqlen_tensor == nullptr, "q_cum_seqlen_tensor != nullptr");
  ORT_RETURN_IF(k_cum_seqlen_tensor == nullptr, "k_cum_seqlen_tensor != nullptr");
  ORT_RETURN_IF(v_cum_seqlen_tensor == nullptr, "v_cum_seqlen_tensor != nullptr");
  ORT_RETURN_IF(bsz_and_seq_shape_tensor == nullptr, "full_shape_tensor != nullptr");

  const int64_t batch_size = q_cum_seqlen_tensor->Shape()[0] - 1;
  ORT_RETURN_IF(q_cum_seqlen_tensor->Shape() != k_cum_seqlen_tensor->Shape(),
                "q_cum_seqlen_tensor->Shape() != k_cum_seqlen_tensor->Shape()");
  ORT_RETURN_IF(q_cum_seqlen_tensor->Shape() != v_cum_seqlen_tensor->Shape(),
                "q_cum_seqlen_tensor->Shape() != v_cum_seqlen_tensor->Shape()");

  // New shape to be expanded to for `token_count` dim.
  const auto* p_bsz_and_seq_shape = bsz_and_seq_shape_tensor->Data<int64_t>();
  TensorShapeVector bsz_and_seq_shape{p_bsz_and_seq_shape, p_bsz_and_seq_shape + bsz_and_seq_shape_tensor->Shape().Size()};
  ORT_RETURN_IF(bsz_and_seq_shape.size() != 2, "bsz_and_seq_shape.size() != 2");
  ORT_RETURN_IF(bsz_and_seq_shape[0] != batch_size, "bsz_and_seq_shape[0] != batch_size");
  const int64_t max_sequence_length = bsz_and_seq_shape[1];

  const auto type = q_tensor->DataType();
  const size_t element_size = type->Size();
  TArray<int> q_perms({0, 2, 1, 3}), k_perms({0, 2, 3, 1}), v_perms({0, 2, 1, 3});

  TArray<int64_t> full_sized_query_shape;
  TArray<int64_t> full_sized_transposed_query_output_shape;
  int64_t query_factor_for_fixed_dims = 1;
  int query_variant_axis_on_output;

  {
    gsl::span<const int64_t> flatten_input_dims = q_tensor->Shape().GetDims();
    full_sized_query_shape.SetSize(flatten_input_dims.size() + 1);
    full_sized_query_shape[0] = bsz_and_seq_shape[0];
    full_sized_query_shape[1] = bsz_and_seq_shape[1];
    for (size_t i = 1; i < flatten_input_dims.size(); ++i) {
      full_sized_query_shape[i + 1] = flatten_input_dims[i];
    }
  }

  PrepareTransposeLaunchInfos(seq_axis_on_input,
                              q_perms,
                              full_sized_query_shape,
                              full_sized_transposed_query_output_shape,
                              query_factor_for_fixed_dims,
                              query_variant_axis_on_output);

  // Transpose Q from [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq and Head are switched, e.g:
  // [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  IAllocatorUniquePtr<T> transposed_query_output_data = GetScratchBuffer<T>(q_tensor->Shape().Size(),
                                                                            context->GetComputeStream());
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           q_cum_seqlen_tensor->Data<int64_t>(),
                                           seq_axis_on_input,
                                           query_variant_axis_on_output,
                                           full_sized_query_shape,
                                           full_sized_transposed_query_output_shape,
                                           query_factor_for_fixed_dims,
                                           q_perms,
                                           q_tensor->Data<T>(),
                                           transposed_query_output_data.get(),
                                           static_cast<size_t>(q_tensor->Shape().Size())));

  TArray<int64_t> full_sized_key_shape;
  TArray<int64_t> full_sized_transposed_key_output_shape;
  int64_t key_factor_for_fixed_dims = 1;
  int key_variant_axis_on_output;

  {
    gsl::span<const int64_t> flatten_input_dims = k_tensor->Shape().GetDims();
    full_sized_key_shape.SetSize(flatten_input_dims.size() + 1);
    full_sized_key_shape[0] = bsz_and_seq_shape[0];
    full_sized_key_shape[1] = bsz_and_seq_shape[1];
    for (size_t i = 1; i < flatten_input_dims.size(); ++i) {
      full_sized_key_shape[i + 1] = flatten_input_dims[i];
    }
  }

  PrepareTransposeLaunchInfos(seq_axis_on_input,
                              k_perms,
                              full_sized_key_shape,
                              full_sized_transposed_key_output_shape,
                              key_factor_for_fixed_dims,
                              key_variant_axis_on_output);

  // Transpose K from [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq1 and `Head, Hidden_per_head` are switched, e.g:
  // [[Head, Hidden_per_head, Seq1], [Head, Hidden_per_head, Seq2], ..., [Head, Hidden_per_head, SeqB]]
  IAllocatorUniquePtr<T> transposed_key_output_data = GetScratchBuffer<T>(k_tensor->Shape().Size(),
                                                                          context->GetComputeStream());
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           k_cum_seqlen_tensor->Data<int64_t>(),
                                           seq_axis_on_input,
                                           key_variant_axis_on_output,
                                           full_sized_key_shape,
                                           full_sized_transposed_key_output_shape,
                                           key_factor_for_fixed_dims,
                                           k_perms,
                                           k_tensor->Data<T>(),
                                           transposed_key_output_data.get(),
                                           static_cast<size_t>(k_tensor->Shape().Size())));

  TArray<int64_t> full_sized_value_shape;
  TArray<int64_t> full_sized_transposed_value_output_shape;
  int64_t value_factor_for_fixed_dims = 1;
  int value_variant_axis_on_output;

  {
    gsl::span<const int64_t> flatten_input_dims = v_tensor->Shape().GetDims();
    full_sized_value_shape.SetSize(flatten_input_dims.size() + 1);
    full_sized_value_shape[0] = bsz_and_seq_shape[0];
    full_sized_value_shape[1] = bsz_and_seq_shape[1];
    for (size_t i = 1; i < flatten_input_dims.size(); ++i) {
      full_sized_value_shape[i + 1] = flatten_input_dims[i];
    }
  }

  PrepareTransposeLaunchInfos(seq_axis_on_input,
                              v_perms,
                              full_sized_value_shape,
                              full_sized_transposed_value_output_shape,
                              value_factor_for_fixed_dims,
                              value_variant_axis_on_output);

  // Transpose V from [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq1 and `Head` are switched, e.g:
  // [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  IAllocatorUniquePtr<T> transposed_value_output_data = GetScratchBuffer<T>(v_tensor->Shape().Size(),
                                                                            context->GetComputeStream());
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           v_cum_seqlen_tensor->Data<int64_t>(),
                                           seq_axis_on_input,
                                           value_variant_axis_on_output,
                                           full_sized_value_shape,
                                           full_sized_transposed_value_output_shape,
                                           value_factor_for_fixed_dims,
                                           v_perms,
                                           v_tensor->Data<T>(),
                                           transposed_value_output_data.get(),
                                           static_cast<size_t>(v_tensor->Shape().Size())));

  // GroupGemm for attention score.
  const int64_t query_head_count = full_sized_query_shape[2];
  const int64_t q_batch_x_head = batch_size * query_head_count;
  const int64_t q_hidden_per_head = full_sized_query_shape[3];
  const int64_t* q_cum_seqlen = q_cum_seqlen_tensor->Data<int64_t>();

  const int64_t key_head_count = full_sized_key_shape[2];
  const int64_t k_batch_x_head = batch_size * key_head_count;
  const int64_t k_hidden_per_head = full_sized_key_shape[3];
  const int64_t* k_cum_seqlen = k_cum_seqlen_tensor->Data<int64_t>();

  const int64_t value_head_count = full_sized_value_shape[2];
  const int64_t v_batch_x_head = batch_size * value_head_count;
  const int64_t v_hidden_per_head = full_sized_value_shape[3];
  // const int64_t* v_cum_seqlen = v_cum_seqlen_tensor->Data<int64_t>();

  ORT_RETURN_IF(query_head_count != key_head_count, "query_head_count != key_head_count");
  ORT_RETURN_IF(query_head_count != value_head_count, "query_head_count != value_head_count");

  ORT_RETURN_IF(q_batch_x_head != k_batch_x_head, "q_batch_x_head != k_batch_x_head");
  ORT_RETURN_IF(q_batch_x_head != v_batch_x_head, "q_batch_x_head != v_batch_x_head");

  ORT_RETURN_IF(q_hidden_per_head != k_hidden_per_head, "q_hidden_per_head != k_hidden_per_head");
  ORT_RETURN_IF(q_hidden_per_head != v_hidden_per_head, "q_hidden_per_head != v_hidden_per_head");

  const int64_t problem_count = q_batch_x_head;
  const int64_t head_count = query_head_count;
  std::vector<std::tuple<int64_t, int64_t, int64_t>> problem_sizes;  // M, N, K
  problem_sizes.reserve(problem_count);

  size_t group_gemm_output_element_count = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(q_batch_x_head); ++i) {
    int32_t batch_index = static_cast<int32_t>(std::ceil(((float)i) / (float(head_count))));
    int64_t q_seq_len = q_cum_seqlen[batch_index + 1] - q_cum_seqlen[batch_index];
    int64_t k_seq_len = k_cum_seqlen[batch_index + 1] - k_cum_seqlen[batch_index];
    // The q seq len and k seq len should be the same as an assumption of the fusion now.
    ORT_RETURN_IF(q_seq_len != k_seq_len, "q_seq_len != k_seq_len");
    problem_sizes.push_back(std::make_tuple(q_seq_len, k_seq_len, q_hidden_per_head));

    // [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
    // X
    // [[Head, Hidden_per_head, Seq1], [Head, Hidden_per_head, Seq2], ..., [Head, Hidden_per_head, SeqB]]
    // ==>
    // [[Head, Seq1, Seq1], [Head, Seq2, Seq2], ..., [Head, SeqB, SeqB]]
    group_gemm_output_element_count += head_count * q_seq_len * k_seq_len;
  }

  typedef typename ToCudaType<T>::MappedType CudaT;

  int64_t token_count = q_tensor->Shape()[0];
  ORT_RETURN_IF(token_count != k_tensor->Shape()[0], "token_count != k_tensor->Shape()[0]");
  ORT_RETURN_IF(token_count != v_tensor->Shape()[0], "token_count != v_tensor->Shape()[0]");

  ORT_RETURN_IF(q_tensor->Shape()[1] != query_head_count, "q_tensor->Shape()[1] != query_head_count");
  ORT_RETURN_IF(k_tensor->Shape()[1] != key_head_count, "k_tensor->Shape()[1] != key_head_count");
  ORT_RETURN_IF(v_tensor->Shape()[1] != value_head_count, "v_tensor->Shape()[1] != value_head_count");

  // const int64_t max_seq_length = bsz_and_seq_shape[1];

  OrtValue group_gemm_output_ortvalue;
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  TensorShapeVector gemm_output_shape_vec{{static_cast<int64_t>(group_gemm_output_element_count)}};
  group_gemm_output_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), gemm_output_shape_vec, alloc);
  T* group_gemm_output_data = group_gemm_output_ortvalue.GetMutable<Tensor>()->template MutableData<T>();

  std::vector<const CudaT*> data_ptr_a_vec(problem_count);
  std::vector<const CudaT*> data_ptr_b_vec(problem_count);
  std::vector<CudaT*> data_ptr_c_vec(problem_count);
  std::vector<CudaT*> data_ptr_d_vec(problem_count);
  for (int64_t i = 0; i < problem_count; ++i) {
    data_ptr_a_vec[i] = reinterpret_cast<const CudaT*>(transposed_query_output_data.get());
    data_ptr_b_vec[i] = reinterpret_cast<const CudaT*>(transposed_key_output_data.get());
    data_ptr_c_vec[i] = reinterpret_cast<CudaT*>(group_gemm_output_data);
    data_ptr_d_vec[i] = reinterpret_cast<CudaT*>(group_gemm_output_data);
  }

  auto ret = contrib::cuda::GroupGemm_Impl<CudaT, true, true>(
      this,
      context->GetComputeStream(),
      problem_sizes,
      problem_count,
      data_ptr_a_vec,
      data_ptr_b_vec,
      data_ptr_c_vec,
      data_ptr_d_vec);

  ORT_RETURN_IF_ERROR(ret);

  // [[Head, Seq1, Seq1], [Head, Seq2, Seq2], ..., [Head, SeqB, SeqB]]
  TensorShapeVector softmax_output_shape_vec{{static_cast<int64_t>(group_gemm_output_element_count)}};
  OrtValue softmax_output_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(),
                                                             softmax_output_shape_vec, alloc);
  Tensor* softmax_output_tensor = softmax_output_ortvalue.GetMutable<Tensor>();
  T* softmax_output_data = softmax_output_tensor->template MutableData<T>();
  // IAllocatorUniquePtr<float> softmax_output_data = GetScratchBuffer<float>(div_output_tensor->Shape().Size(),
  //                                                                          context->GetComputeStream());
  // TODO(pengwa): implement var-length softmax.
  // Status status = SoftMaxComputeHelper<T, T, false>(context->GetComputeStream(), softmax_output_data,
  //                                                   softmax_output_tensor->Shape(), softmax_output_data,
  //                                                   1 /*reduction on the 2nd dim*/);

  Status status = SoftMaxVarLengthComputeHelper<T, T, false>(context->GetComputeStream(),
                                                             softmax_output_data,
                                                             max_sequence_length,
                                                             q_cum_seqlen_tensor->Data<int64_t>(),
                                                             batch_size,
                                                             head_count,
                                                             softmax_output_data);

  ORT_RETURN_IF_ERROR(status);

  TensorShape mask_shape({static_cast<int64_t>(group_gemm_output_element_count)});
  Tensor* mask = context->Output(0, mask_shape);
  // Get the ratio_data
  float ratio_data = default_ratio_;
  // auto ratio = context->Input<Tensor>(1);
  // if (ratio) {
  //   utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
  //   t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  // }
  int64_t mask_element_count = mask_shape.Size();
  IAllocatorUniquePtr<void> temp_mask_buffer{};  // buffer to use if mask is not provided
  void* const mask_data = [this, mask_element_count, mask, &temp_mask_buffer, context]() {
    if (mask) return mask->MutableDataRaw();
    temp_mask_buffer =
        GetScratchBuffer<void>(mask_element_count * sizeof(bool), context->GetComputeStream());
    return temp_mask_buffer.get();
  }();

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(softmax_output_tensor->GetElementType());
  t_disp.Invoke<DropoutComputeImpl>(GetDeviceProp(), Stream(context), mask_element_count, mask_element_count,
                                    ratio_data, generator, *softmax_output_tensor, *softmax_output_tensor,
                                    mask_data, false /*UseBitmask*/);

  // GroupGemm for attention score X Value.

  // inputs: [[Head, Seq1, Seq1], [Head, Seq2, Seq2], ..., [Head, SeqB, SeqB]] x [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  // output: [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  // OrtValue group_gemm_output_ortvalue;
  // TensorShapeVector gemm_output_shape_vec(q_tensor->Shape().GetDims());
  // group_gemm_output_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), gemm_output_shape_vec, alloc);
  // float* group_gemm_output_data = group_gemm_output_ortvalue.GetMutable<Tensor>()->template MutableData<float>();

  IAllocatorUniquePtr<T> weighted_sum_output_data = GetScratchBuffer<T>(q_tensor->Shape().Size(),
                                                                        context->GetComputeStream());

  const int64_t weighted_sum_problem_count = q_batch_x_head;
  std::vector<std::tuple<int64_t, int64_t, int64_t>> weighted_sum_problem_sizes;  // M, N, K
  weighted_sum_problem_sizes.reserve(weighted_sum_problem_count);

  std::vector<const CudaT*> weighted_sum_data_ptr_a_vec(weighted_sum_problem_count);
  std::vector<const CudaT*> weighted_sum_data_ptr_b_vec(weighted_sum_problem_count);
  std::vector<CudaT*> weighted_sum_data_ptr_c_vec(weighted_sum_problem_count);
  std::vector<CudaT*> weighted_sum_data_ptr_d_vec(weighted_sum_problem_count);
  const T* value_data = q_tensor->Data<T>();
  int64_t lhs_input_offset = 0;
  int64_t rhs_input_offset = 0;
  int64_t output_offset = 0;
  for (int64_t i = 0; i < weighted_sum_problem_count; ++i) {
    int32_t batch_index = static_cast<int32_t>(std::ceil(((float)i) / (float(head_count))));
    int64_t seq_len = q_cum_seqlen[batch_index + 1] - q_cum_seqlen[batch_index];

    weighted_sum_data_ptr_a_vec[i] = reinterpret_cast<const CudaT*>(softmax_output_data + lhs_input_offset);
    weighted_sum_data_ptr_b_vec[i] = reinterpret_cast<const CudaT*>(value_data + rhs_input_offset);
    weighted_sum_data_ptr_c_vec[i] = reinterpret_cast<CudaT*>(weighted_sum_output_data.get() + output_offset);
    weighted_sum_data_ptr_d_vec[i] = reinterpret_cast<CudaT*>(weighted_sum_output_data.get() + output_offset);

    lhs_input_offset += head_count * seq_len * seq_len;
    rhs_input_offset += head_count * seq_len * v_hidden_per_head;
    output_offset += head_count * seq_len * v_hidden_per_head;
  }

  auto weighted_sum_ret = contrib::cuda::GroupGemm_Impl<CudaT, true, true>(
      this,
      context->GetComputeStream(),
      weighted_sum_problem_sizes,
      weighted_sum_problem_count,
      weighted_sum_data_ptr_a_vec,
      weighted_sum_data_ptr_b_vec,
      weighted_sum_data_ptr_c_vec,
      weighted_sum_data_ptr_d_vec);

  ORT_RETURN_IF_ERROR(weighted_sum_ret);

  // input: [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  // output: [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]

  TArray<int> s_perms({0, 2, 1, 3});
  constexpr static int seq_axis_on_score = 2;
  TArray<int64_t> full_sized_score_shape;
  TArray<int64_t> full_sized_transposed_score_output_shape;
  int64_t score_factor_for_fixed_dims = 1;
  int score_variant_axis_on_output;

  {
    gsl::span<const int64_t> flatten_input_dims = v_tensor->Shape().GetDims();
    full_sized_score_shape.SetSize(flatten_input_dims.size() + 1);
    full_sized_score_shape[0] = bsz_and_seq_shape[0];
    full_sized_score_shape[1] = flatten_input_dims[1];
    full_sized_score_shape[2] = bsz_and_seq_shape[1];
    full_sized_score_shape[3] = flatten_input_dims[2];
  }

  PrepareTransposeLaunchInfos(seq_axis_on_score,
                              s_perms,
                              full_sized_score_shape,
                              full_sized_transposed_score_output_shape,
                              score_factor_for_fixed_dims,
                              score_variant_axis_on_output);

  // Transpose Score from [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq and Head are switched, e.g:
  // [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  IAllocatorUniquePtr<T> transposed_score_output_data = GetScratchBuffer<T>(v_tensor->Shape().Size(),
                                                                            context->GetComputeStream());
  constexpr static int value_variant_axis_on_score_output = 2;
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           v_cum_seqlen_tensor->Data<int64_t>(),
                                           seq_axis_on_score,
                                           value_variant_axis_on_score_output,
                                           full_sized_score_shape,
                                           full_sized_transposed_score_output_shape,
                                           score_factor_for_fixed_dims,
                                           s_perms,
                                           weighted_sum_output_data.get(),
                                           transposed_score_output_data.get(),
                                           static_cast<size_t>(v_tensor->Shape().Size())));

  return Status::OK();
}

REGISTER_KERNEL_TYPED(float);
REGISTER_KERNEL_TYPED(double);
REGISTER_KERNEL_TYPED(MLFloat16);

}  // namespace cuda
}  // namespace onnxruntime
