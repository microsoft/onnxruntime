// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/transformer/unfused_dot_prod_attention_var_len.h"
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
#include "orttraining/training_ops/cuda/transformer/unfused_dot_prod_attention_var_len_impl.h"

#include "contrib_ops/cuda/math/group_gemm.h"
#include "contrib_ops/cuda/math/group_gemm_impl.h"
// #include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

#include "core/providers/cuda/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    UnfusedScaledDotProductAttentionVariableSeqlen,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("M", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int64_t>(),
                             })
        .TypeConstraint("TI", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<int64_t>(),
                              }),
    UnfusedScaledDotProductAttentionVariableSeqlen);

namespace {

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

Status BinaryElementwiseShouldBroadcastPrepare(OpKernelContext*,
                                               BinaryElementwisePreparation* p,
                                               const Tensor* lhs_tensor,
                                               const Tensor* rhs_tensor,
                                               Tensor* output_tensor) {
  // auto lhs_tensor = context->Input<Tensor>(0);
  // auto rhs_tensor = context->Input<Tensor>(1);
  // const auto& lhs_shape = lhs_tensor->Shape();
  // const auto& rhs_shape = rhs_tensor->Shape();

  // TensorShape output_shape;
  // ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), lhs_shape, rhs_shape, output_shape));
  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(lhs_tensor, rhs_tensor, output_tensor, p));

  return Status::OK();
}

}  // namespace

Status UnfusedScaledDotProductAttentionVariableSeqlen::ComputeInternal(OpKernelContext* context) const {
  const Tensor* q_tensor = context->Input<Tensor>(0);
  const Tensor* k_tensor = context->Input<Tensor>(1);
  const Tensor* v_tensor = context->Input<Tensor>(2);
  const Tensor* q_cum_seqlen_tensor = context->Input<Tensor>(3);
  const Tensor* k_cum_seqlen_tensor = context->Input<Tensor>(4);
  const Tensor* v_cum_seqlen_tensor = context->Input<Tensor>(5);

  const Tensor* mask_tensor = context->Input<Tensor>(6);
  const Tensor* full_shape_tensor = context->Input<Tensor>(7);

  ORT_RETURN_IF(q_tensor == nullptr, "q_tensor != nullptr");
  ORT_RETURN_IF(k_tensor == nullptr, "k_tensor != nullptr");
  ORT_RETURN_IF(v_tensor == nullptr, "v_tensor != nullptr");
  ORT_RETURN_IF(q_cum_seqlen_tensor == nullptr, "q_cum_seqlen_tensor != nullptr");
  ORT_RETURN_IF(k_cum_seqlen_tensor == nullptr, "k_cum_seqlen_tensor != nullptr");
  ORT_RETURN_IF(v_cum_seqlen_tensor == nullptr, "v_cum_seqlen_tensor != nullptr");
  ORT_RETURN_IF(mask_tensor == nullptr, "mask_tensor != nullptr");
  ORT_RETURN_IF(full_shape_tensor == nullptr, "full_shape_tensor != nullptr");

  // new shape to be expanded to
  const auto* p_full_shape = full_shape_tensor->Data<int64_t>();
  TensorShapeVector full_shape_dims{p_full_shape, p_full_shape + full_shape_tensor->Shape().Size()};

  // 3 Transpose Nodes
  // K: 0, 2, 3, 1
  // Q: 0, 2, 1, 3
  // V: 0, 2, 1, 3

  std::vector<size_t> k_perms({0, 2, 3, 1}), q_perms({0, 2, 1, 3}), v_perms({0, 2, 1, 3});
  TArray<int> q_perms_tarray(q_perms.size()), k_perms_tarray(k_perms.size()), v_perms_tarray(v_perms.size());
  for (size_t i = 0; i < q_perms.size(); ++i) {
    q_perms_tarray[i] = q_perms[i];
    k_perms_tarray[i] = k_perms[i];
    v_perms_tarray[i] = v_perms[i];
  }

  int variant_axis_on_input = 1;
  // int k_variant_axis_on_output = k_perms[variant_axis_on_input];
  int q_variant_axis_on_output = q_perms[variant_axis_on_input];
  // int k_variant_axis_on_output = k_perms[variant_axis_on_input];

  const int transpose_input_dim_size = 4;

  TArray<int64_t> input_shape(ToConstSpan(full_shape_dims));

  TArray<int64_t> k_output_shape(transpose_input_dim_size), q_output_shape(transpose_input_dim_size),
      v_output_shape(transpose_input_dim_size);

  TensorShapeVector k_output_shape_vec;
  k_output_shape_vec.reserve(transpose_input_dim_size);

  // TArray<int> k_reverse_perms({transpose_input_dim_size}), q_reverse_perms({transpose_input_dim_size}),
  //     v_reverse_perms({transpose_input_dim_size});
  int64_t factor_for_fixed_dims = 1;
  for (int i = 0; i < transpose_input_dim_size; ++i) {
    k_output_shape[i] = full_shape_dims[k_perms[i]];
    k_output_shape_vec[i] = full_shape_dims[k_perms[i]];
    q_output_shape[i] = full_shape_dims[q_perms[i]];
    v_output_shape[i] = full_shape_dims[v_perms[i]];

    if (i != variant_axis_on_input) {
      factor_for_fixed_dims *= full_shape_dims[i];
    }
  }

  size_t element_size = 4;

  // Transpose Q from [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq and Head are switched, e.g:
  // [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  const float* q_input_data = q_tensor->Data<float>();
  IAllocatorUniquePtr<float> q_output_data = GetScratchBuffer<float>(q_tensor->Shape().Size(),
                                                                     context->GetComputeStream());
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           q_cum_seqlen_tensor->Data<int64_t>(),
                                           variant_axis_on_input,
                                           q_variant_axis_on_output,
                                           input_shape,
                                           q_output_shape,
                                           factor_for_fixed_dims,
                                           q_perms_tarray,
                                           q_input_data,
                                           q_output_data.get(),
                                           static_cast<size_t>(q_tensor->Shape().Size())));

  // Transpose K from [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq1 and `Head, Hidden_per_head` are switched, and Seq1 is changed to MaxSeq
  // (filling zero for non-existence), e.g:
  // [[Head, Hidden_per_head, MaxSeq], [Head, Hidden_per_head, MaxSeq], ..., [Head, Hidden_per_head, MaxSeq]]
  const float* k_input_data = k_tensor->Data<float>();
  TensorShape k_output_tensor_shape(full_shape_dims);
  IAllocatorUniquePtr<float> k_output_data = GetScratchBuffer<float>(k_output_tensor_shape.Size(),
                                                                     context->GetComputeStream());
  TensorPitches new_output_strides(k_output_tensor_shape);
  TArray<fast_divmod> output_strides(full_shape_dims.size());
  for (auto i = 0; i < full_shape_dims.size(); i++) {
    output_strides[i] = fast_divmod(gsl::narrow_cast<int>(new_output_strides[i]));
  }
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           k_cum_seqlen_tensor->Data<int64_t>(),
                                           variant_axis_on_input,
                                           //  q_variant_axis_on_output,
                                           input_shape,
                                           //  k_output_shape,
                                           factor_for_fixed_dims,
                                           k_perms_tarray,
                                           k_input_data,
                                           output_strides,
                                           k_output_data.get(),
                                           static_cast<size_t>(k_output_tensor_shape.Size())));

  // Transpose V from [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq1 and `Head` are switched, and Seq1 is changed to MaxSeq
  // (filling zero for non-existence), e.g:
  // [[Head, MaxSeq, Hidden_per_head], [Head, MaxSeq, Hidden_per_head], ..., [Head, MaxSeq, Hidden_per_head]]
  const float* v_input_data = v_tensor->Data<float>();
  TensorShape v_output_tensor_shape(full_shape_dims);
  IAllocatorUniquePtr<float> v_output_data = GetScratchBuffer<float>(v_output_tensor_shape.Size(),
                                                                     context->GetComputeStream());
  TensorPitches v_new_output_strides(v_output_tensor_shape);
  TArray<fast_divmod> v_output_strides(full_shape_dims.size());
  for (auto i = 0; i < full_shape_dims.size(); i++) {
    v_output_strides[i] = fast_divmod(gsl::narrow_cast<int>(v_new_output_strides[i]));
  }
  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           v_cum_seqlen_tensor->Data<int64_t>(),
                                           variant_axis_on_input,
                                           //  q_variant_axis_on_output,
                                           input_shape,
                                           //  k_output_shape,
                                           factor_for_fixed_dims,
                                           v_perms_tarray,
                                           v_input_data,
                                           v_output_strides,
                                           v_output_data.get(),
                                           static_cast<size_t>(v_output_tensor_shape.Size())));

  // GroupGemm for attention score.
  size_t num_batch = q_cum_seqlen_tensor->Shape().Size() - 1;
  size_t head = input_shape[2];
  size_t q_batch_x_head = num_batch * head;
  size_t q_hidden_per_head = input_shape[3];
  const int64_t* q_cum_seqlen = q_cum_seqlen_tensor->Data<int64_t>();

  size_t problem_count = q_batch_x_head;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes;
  problem_sizes.reserve(problem_count);

  for (int64_t i = 0; i < static_cast<int64_t>(q_batch_x_head); ++i) {
    int32_t batch_index = static_cast<int32_t>(std::ceil(((float)i) / (float(head))));
    int64_t q_seq_len = q_cum_seqlen[batch_index + 1] - q_cum_seqlen[batch_index];
    problem_sizes.push_back(cutlass::gemm::GemmCoord(q_seq_len, q_hidden_per_head, k_output_shape[3]));
  }

  typedef typename ToCudaType<float>::MappedType CudaT;

  CudaAsyncBuffer<cutlass::gemm::GemmCoord> problem_sizes_device(this, problem_count);
  CudaAsyncBuffer<int64_t> lda(this, problem_count);
  CudaAsyncBuffer<int64_t> ldb(this, problem_count);
  CudaAsyncBuffer<int64_t> ldc(this, problem_count);
  CudaAsyncBuffer<int64_t> ldd(this, problem_count);

  CudaAsyncBuffer<CudaT*> data_ptr_a(this, problem_count);
  CudaAsyncBuffer<CudaT*> data_ptr_b(this, problem_count);
  CudaAsyncBuffer<CudaT*> data_ptr_c(this, problem_count);
  CudaAsyncBuffer<CudaT*> data_ptr_d(this, problem_count);

  gsl::span<cutlass::gemm::GemmCoord> problem_sizes_span = problem_sizes_device.CpuSpan();
  gsl::span<int64_t> lda_span = lda.CpuSpan();
  gsl::span<int64_t> ldb_span = ldb.CpuSpan();
  gsl::span<int64_t> ldc_span = ldc.CpuSpan();
  gsl::span<int64_t> ldd_span = ldd.CpuSpan();

  gsl::span<CudaT*> data_ptr_a_span = data_ptr_a.CpuSpan();
  gsl::span<CudaT*> data_ptr_b_span = data_ptr_b.CpuSpan();
  gsl::span<CudaT*> data_ptr_c_span = data_ptr_c.CpuSpan();
  gsl::span<CudaT*> data_ptr_d_span = data_ptr_d.CpuSpan();

  int64_t token_count = q_tensor->Shape()[0];
  int64_t head_count = q_tensor->Shape()[1];
  // int64_t hidden_size_per_head = q_tensor->Shape()[2];
  int64_t max_seq_length = full_shape_dims[1];
  // q_tensor->Shape()[0] * q_tensor->Shape()[1] * q_cum_seqlen[num_batch];
  size_t group_gemm_ret_size = token_count * head_count * max_seq_length;
  // IAllocatorUniquePtr<float> group_gemm_output_data = GetScratchBuffer<float>(group_gemm_ret_size,
  //                                                                             context->GetComputeStream());

  OrtValue group_gemm_output_ortvalue;
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  TensorShapeVector gemm_output_shape_vec{static_cast<int64_t>(group_gemm_ret_size)};
  group_gemm_output_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), gemm_output_shape_vec, alloc);
  float* group_gemm_output_data = group_gemm_output_ortvalue.GetMutable<Tensor>()->template MutableData<float>();

  for (size_t i = 0; i < problem_count; ++i) {
    auto& problem = problem_sizes[i];
    problem_sizes_span[i] = problem;
    data_ptr_a_span[i] = const_cast<CudaT*>(reinterpret_cast<const CudaT*>(q_output_data.get()));
    data_ptr_b_span[i] = const_cast<CudaT*>(reinterpret_cast<const CudaT*>(k_output_data.get()));
    data_ptr_c_span[i] = reinterpret_cast<CudaT*>(group_gemm_output_data);
    data_ptr_d_span[i] = reinterpret_cast<CudaT*>(group_gemm_output_data);
  }

  contrib::cuda::GenerateLdaLdbLdcLdd<true>(problem_sizes, lda_span, ldb_span, ldc_span, ldd_span);

  ORT_RETURN_IF_ERROR(problem_sizes_device.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(lda.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(ldb.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(ldc.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(ldd.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_a.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_b.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_c.CopyToGpu(context->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_d.CopyToGpu(context->GetComputeStream()));

  auto ret = contrib::cuda::GroupGemm_Impl<CudaT, true, true>(
      this,
      context->GetComputeStream(),
      problem_sizes,
      problem_sizes_device.GpuPtr(),
      problem_count,
      lda.GpuPtr(),
      ldb.GpuPtr(),
      ldc.GpuPtr(),
      ldd.GpuPtr(),
      data_ptr_a.GpuPtr(),
      data_ptr_b.GpuPtr(),
      data_ptr_c.GpuPtr(),
      data_ptr_d.GpuPtr());

  ORT_RETURN_IF_ERROR(ret);

  OrtValue div_factor_ortvalue;
  // ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  TensorShapeVector div_factor_shape_vec{};
  div_factor_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), div_factor_shape_vec, alloc);
  float* div_factor_data = (*div_factor_ortvalue.GetMutable<Tensor>()).template MutableData<float>();
  *div_factor_data = 8.0f;

  OrtValue div_output_ortvalue;
  // ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  TensorShapeVector div_output_shape_vec{static_cast<int64_t>(group_gemm_ret_size)};
  div_output_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), div_output_shape_vec, alloc);

  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(BinaryElementwiseShouldBroadcastPrepare(
      context,
      &prepare,
      group_gemm_output_ortvalue.GetMutable<Tensor>(),
      div_factor_ortvalue.GetMutable<Tensor>(),
      div_output_ortvalue.GetMutable<Tensor>()));

  auto cuda_stream = context->GetComputeStream()
                         ? static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle())
                         : nullptr;
  Impl_Div<CudaT>(
      cuda_stream,
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const CudaT*>(group_gemm_output_ortvalue.GetMutable<Tensor>()->template MutableData<float>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<CudaT*>(div_factor_ortvalue.GetMutable<Tensor>()->template MutableData<float>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H, prepare.fdm_C,
      reinterpret_cast<CudaT*>(div_output_ortvalue.GetMutable<Tensor>()->template MutableData<float>()),
      static_cast<int64_t>(group_gemm_ret_size));

  // temp_X = cuda_ep.GetScratchBuffer<float>(input_count, ort_stream, WaitCudaNotificationOnDevice);
  Tensor* div_output_tensor = div_output_ortvalue.GetMutable<Tensor>();
  // IAllocatorUniquePtr<float> div_outpput_casted_data = GetScratchBuffer<float>(div_output_tensor->Shape().Size(),
  //                                                                              context->GetComputeStream());
  OrtValue div_output_casted_ortvalue;
  // todo: change data type
  div_output_casted_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), div_output_shape_vec, alloc);
  Tensor* div_output_casted_tensor = div_output_casted_ortvalue.GetMutable<Tensor>();
  Impl_Cast<CudaT, float>(cuda_stream, reinterpret_cast<const CudaT*>(div_output_tensor->template MutableData<float>()),
                          div_output_casted_tensor->template MutableData<float>(), div_output_tensor->Shape().Size());

  // Be noted: missing an add operator now.
  TensorShapeVector softmax_output_shape_vec{{token_count * head_count, max_seq_length}};
  OrtValue softmax_output_ortvalue = AllocateTensorInMLValue(div_output_casted_tensor->DataType(),
                                                             softmax_output_shape_vec, alloc);
  Tensor* softmax_output_tensor = softmax_output_ortvalue.GetMutable<Tensor>();
  float* softmax_output_data = softmax_output_tensor->template MutableData<float>();
  // IAllocatorUniquePtr<float> softmax_output_data = GetScratchBuffer<float>(div_output_tensor->Shape().Size(),
  //                                                                          context->GetComputeStream());
  Status status = SoftMaxComputeHelper<float, float, false>(cuda_stream, softmax_output_data,
                                                            softmax_output_tensor->Shape(), softmax_output_data,
                                                            1 /*reduction on the 2nd dim*/);

  ORT_RETURN_IF_ERROR(status);

  TensorShape mask_shape({token_count * head_count * max_seq_length});
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

  // todo: specifiy the type.
  OrtValue dropout_output_casted_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), div_output_shape_vec, alloc);
  Tensor* dropout_output_casted_tensor = dropout_output_casted_ortvalue.GetMutable<Tensor>();
  Impl_Cast<CudaT, float>(cuda_stream, reinterpret_cast<const CudaT*>(softmax_output_tensor->template MutableData<float>()),
                          dropout_output_casted_tensor->template MutableData<float>(), softmax_output_tensor->Shape().Size());

  OrtValue group_gemm2_output_ortvalue;
  {
    // GroupGemm for attention score X Value.
    size_t num_batch = q_cum_seqlen_tensor->Shape().Size() - 1;
    size_t head = input_shape[2];
    size_t q_batch_x_head = num_batch * head;
    size_t q_hidden_per_head = input_shape[3];
    const int64_t* q_cum_seqlen = q_cum_seqlen_tensor->Data<int64_t>();

    size_t problem_count = q_batch_x_head;
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    problem_sizes.reserve(problem_count);

    for (int64_t i = 0; i < static_cast<int64_t>(q_batch_x_head); ++i) {
      int32_t batch_index = static_cast<int32_t>(std::ceil(((float)i) / (float(head))));
      int64_t q_seq_len = q_cum_seqlen[batch_index + 1] - q_cum_seqlen[batch_index];
      problem_sizes.push_back(cutlass::gemm::GemmCoord(q_seq_len, q_hidden_per_head, max_seq_length));
    }

    typedef typename ToCudaType<float>::MappedType CudaT;

    CudaAsyncBuffer<cutlass::gemm::GemmCoord> problem_sizes_device(this, problem_count);
    CudaAsyncBuffer<int64_t> lda(this, problem_count);
    CudaAsyncBuffer<int64_t> ldb(this, problem_count);
    CudaAsyncBuffer<int64_t> ldc(this, problem_count);
    CudaAsyncBuffer<int64_t> ldd(this, problem_count);

    CudaAsyncBuffer<CudaT*> data_ptr_a(this, problem_count);
    CudaAsyncBuffer<CudaT*> data_ptr_b(this, problem_count);
    CudaAsyncBuffer<CudaT*> data_ptr_c(this, problem_count);
    CudaAsyncBuffer<CudaT*> data_ptr_d(this, problem_count);

    gsl::span<cutlass::gemm::GemmCoord> problem_sizes_span = problem_sizes_device.CpuSpan();
    gsl::span<int64_t> lda_span = lda.CpuSpan();
    gsl::span<int64_t> ldb_span = ldb.CpuSpan();
    gsl::span<int64_t> ldc_span = ldc.CpuSpan();
    gsl::span<int64_t> ldd_span = ldd.CpuSpan();

    gsl::span<CudaT*> data_ptr_a_span = data_ptr_a.CpuSpan();
    gsl::span<CudaT*> data_ptr_b_span = data_ptr_b.CpuSpan();
    gsl::span<CudaT*> data_ptr_c_span = data_ptr_c.CpuSpan();
    gsl::span<CudaT*> data_ptr_d_span = data_ptr_d.CpuSpan();

    size_t group_gemm_ret_size = token_count * head_count * q_hidden_per_head;

    // AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    TensorShapeVector gemm_output_shape_vec{static_cast<int64_t>(group_gemm_ret_size)};
    group_gemm2_output_ortvalue = AllocateTensorInMLValue(q_tensor->DataType(), gemm_output_shape_vec, alloc);
    float* group_gemm_output_data = group_gemm2_output_ortvalue.GetMutable<Tensor>()->template MutableData<float>();

    for (size_t i = 0; i < problem_count; ++i) {
      auto& problem = problem_sizes[i];
      problem_sizes_span[i] = problem;
      data_ptr_a_span[i] = const_cast<CudaT*>(reinterpret_cast<const CudaT*>(q_output_data.get()));
      data_ptr_b_span[i] = const_cast<CudaT*>(reinterpret_cast<const CudaT*>(k_output_data.get()));
      data_ptr_c_span[i] = reinterpret_cast<CudaT*>(group_gemm_output_data);
      data_ptr_d_span[i] = reinterpret_cast<CudaT*>(group_gemm_output_data);
    }

    contrib::cuda::GenerateLdaLdbLdcLdd<true>(problem_sizes, lda_span, ldb_span, ldc_span, ldd_span);

    ORT_RETURN_IF_ERROR(problem_sizes_device.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(lda.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(ldb.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(ldc.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(ldd.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(data_ptr_a.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(data_ptr_b.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(data_ptr_c.CopyToGpu(context->GetComputeStream()));
    ORT_RETURN_IF_ERROR(data_ptr_d.CopyToGpu(context->GetComputeStream()));

    auto ret = contrib::cuda::GroupGemm_Impl<CudaT, true, true>(
        this,
        context->GetComputeStream(),
        problem_sizes,
        problem_sizes_device.GpuPtr(),
        problem_count,
        lda.GpuPtr(),
        ldb.GpuPtr(),
        ldc.GpuPtr(),
        ldd.GpuPtr(),
        data_ptr_a.GpuPtr(),
        data_ptr_b.GpuPtr(),
        data_ptr_c.GpuPtr(),
        data_ptr_d.GpuPtr());

    ORT_RETURN_IF_ERROR(ret);
  }

  // Transpose 2nd group gemm output from [[Head, Seq1, Hidden_per_head], [Head, Seq2, Hidden_per_head], ..., [Head, SeqB, Hidden_per_head]]
  // B is batch size; Head is head count; Hidden_per_head is hidden size per head.
  // After transpose, Seq and Head are switched, e.g:
  // [[Seq1, Head, Hidden_per_head], [Seq2, Head, Hidden_per_head], ..., [SeqB, Head, Hidden_per_head]]
  Tensor* group_gemm2_output_tensor = group_gemm2_output_ortvalue.GetMutable<Tensor>();
  const float* last_transpose_input_data = group_gemm2_output_tensor->Data<float>();
  IAllocatorUniquePtr<float> last_transpose_output_data = GetScratchBuffer<float>(group_gemm2_output_tensor->Shape().Size(),
                                                                                  context->GetComputeStream());

  std::vector<size_t> last_trans_perms({0, 2, 1, 3});
  TArray<int> last_trans_perms_tarray(last_trans_perms.size());
  for (size_t i = 0; i < last_trans_perms.size(); ++i) {
    last_trans_perms_tarray[i] = last_trans_perms[i];
  }

  TArray<int64_t> last_trans_input_shape(group_gemm2_output_tensor->Shape().NumDimensions());
  for (auto i = 0; i < group_gemm2_output_tensor->Shape().NumDimensions(); ++i) {
    last_trans_input_shape[i] = group_gemm2_output_tensor->Shape()[i];
  }
  TArray<int64_t> last_trans_output_shape(full_shape_dims);

  // ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
  //                                          element_size,
  //                                          q_cum_seqlen_tensor->Data<int64_t>(),
  //                                          variant_axis_on_input,
  //                                          q_variant_axis_on_output,
  //                                          input_shape,
  //                                          q_output_shape,
  //                                          factor_for_fixed_dims,
  //                                          q_perms_tarray,
  //                                          q_input_data,
  //                                          q_output_data.get(),
  //                                          static_cast<size_t>(q_tensor->Shape().Size())));

  ORT_RETURN_IF_ERROR(LaunchGroupTranspose(Stream(context),
                                           element_size,
                                           q_cum_seqlen_tensor->Data<int64_t>(),
                                           2 /*variant_axis_on_input*/,
                                           1 /*variant_axis_on_output*/,
                                           last_trans_input_shape,
                                           last_trans_output_shape,
                                           factor_for_fixed_dims,
                                           last_trans_perms_tarray,
                                           last_transpose_input_data,
                                           last_transpose_output_data.get(),
                                           static_cast<size_t>(group_gemm2_output_tensor->Shape().Size())));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
