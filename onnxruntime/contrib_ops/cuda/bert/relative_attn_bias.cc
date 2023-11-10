// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "relative_attn_bias.h"
#include "core/common/safeint.h"
#include "relative_attn_bias_impl.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      RelativePositionBias,                                       \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RelPosAttnBias<T>);                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GatedRelativePositionBias,                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GatedRelativePositionBias<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
RelPosAttnBias<T>::RelPosAttnBias(const OpKernelInfo& info) : CudaKernel(info) {
  is_bidirectional_ = info.GetAttrOrDefault<int64_t>("is_bidirectional", 0) == 1;

  int64_t max_distance = 0;
  ORT_ENFORCE(info.GetAttr("max_distance", &max_distance).IsOK() && max_distance > 0);
  max_distance_ = static_cast<int>(max_distance);
}

template <typename T>
Status RelPosAttnBias<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* bias_table = context->Input<Tensor>(0);
  const Tensor* query_length = context->Input<Tensor>(1);
  const Tensor* key_length = context->Input<Tensor>(2);

  const auto& bias_table_dims = bias_table->Shape().GetDims();
  const int64_t num_buckets = bias_table_dims[0];
  const int64_t num_heads = bias_table_dims[1];

  const int64_t query_len = *query_length->Data<int64_t>();
  const int64_t key_len = *key_length->Data<int64_t>();

  if (query_len != key_len) {
    ORT_THROW("Relative position bias currently only support query length equal to key length in Self Attention.");
  }

  Tensor* output = context->Output(0, {1, num_heads, query_len, key_len});

  typedef typename ToCudaType<T>::MappedType CudaT;

  auto& device_prop = GetDeviceProp();
  return LaunchRelPosAttnBiasKernel<CudaT>(Stream(context),
                                           reinterpret_cast<CudaT*>(output->template MutableData<T>()),
                                           reinterpret_cast<const CudaT*>(bias_table->template Data<T>()),
                                           static_cast<int>(num_heads),
                                           static_cast<int>(query_len),
                                           static_cast<int>(num_buckets),
                                           max_distance_,
                                           is_bidirectional_,
                                           device_prop.maxThreadsPerBlock);
}

template <typename T>
GatedRelativePositionBias<T>::GatedRelativePositionBias(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = SafeInt<int>(num_heads);
}

template <typename T>
Status GatedRelativePositionBias<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor& query_tensor = *context->Input<Tensor>(0);
  const Tensor& query_bias_tensor = *context->Input<Tensor>(1);
  const Tensor& rel_pos_tensor = *context->Input<Tensor>(2);
  const Tensor& weight_tensor = *context->Input<Tensor>(3);
  const Tensor& bias_tensor = *context->Input<Tensor>(4);
  const Tensor& eco_a_tensor = *context->Input<Tensor>(5);
  const Tensor* token_offset_tensor = context->Input<Tensor>(6);

  const auto& query_dims = query_tensor.Shape().GetDims();

  bool is_padding_removed = (token_offset_tensor != nullptr);
  ORT_ENFORCE(query_dims.size() == (is_padding_removed ? 2 : 3));
  int hidden_size = static_cast<int>(query_dims[query_dims.size() - 1]);

  ORT_ENFORCE(hidden_size > 0);
  ORT_ENFORCE(hidden_size % num_heads_ == 0);
  const auto head_size = SafeInt<int>(hidden_size / num_heads_);

  int batch_size;
  int seq_len;
  int token_count = 0;
  if (is_padding_removed) {
    const auto& token_offset_dims = token_offset_tensor->Shape().GetDims();
    batch_size = SafeInt<int>(token_offset_dims[0]);
    seq_len = SafeInt<int>(token_offset_dims[1]);
    token_count = SafeInt<int>(query_dims[0]);
  } else {
    batch_size = SafeInt<int>(query_dims[0]);
    seq_len = SafeInt<int>(query_dims[1]);
  }

  ORT_ENFORCE(query_bias_tensor.Shape().NumDimensions() == 1);
  ORT_ENFORCE(query_bias_tensor.Shape()[0] == hidden_size);

  const auto& rel_pos_dims = rel_pos_tensor.Shape().GetDims();
  ORT_ENFORCE(rel_pos_dims.size() == 4);
  ORT_ENFORCE(rel_pos_dims[0] == 1);
  ORT_ENFORCE(rel_pos_dims[1] == num_heads_);
  ORT_ENFORCE(rel_pos_dims[2] == seq_len);
  ORT_ENFORCE(rel_pos_dims[3] == seq_len);

  const auto& weight_dims = weight_tensor.Shape().GetDims();
  ORT_ENFORCE(weight_dims.size() == 2);
  ORT_ENFORCE(weight_dims[0] == head_size);
  ORT_ENFORCE((weight_dims[1] > 0) && (weight_dims[1] % 2 == 0));

  ORT_ENFORCE(bias_tensor.Shape().NumDimensions() == 1);
  ORT_ENFORCE(bias_tensor.Shape()[0] == weight_dims[1]);

  const auto D = SafeInt<int>(weight_dims[1]);

  const auto& eco_a_dims = eco_a_tensor.Shape().GetDims();
  ORT_ENFORCE(eco_a_dims.size() == 4);
  ORT_ENFORCE(eco_a_dims[0] == 1);
  ORT_ENFORCE(eco_a_dims[1] == num_heads_);
  ORT_ENFORCE(eco_a_dims[2] == 1);
  ORT_ENFORCE(eco_a_dims[3] == 1);

  Tensor* output = context->Output(0, {batch_size, num_heads_, seq_len, seq_len});

  auto& device_prop = GetDeviceProp();
  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto BNS = batch_size * num_heads_ * seq_len;
  const size_t elements_in_query = (size_t)BNS * (size_t)head_size;
  const size_t elements_after_gemm = (size_t)BNS * (size_t)D;
  bool reuse_output = (seq_len >= D);
  size_t workspace_size = sizeof(T) * (elements_in_query + (reuse_output ? (size_t)0 : elements_after_gemm));
  auto workspace = GetScratchBuffer<void>(workspace_size, context->GetComputeStream());

  cudaStream_t stream = Stream(context);
  if (!is_padding_removed) {
    // format 1: BxSx(NH * total_matrix) => matrix_to_transpose * (BxNxSxH)
    constexpr int format = 1;
    constexpr int total_maxtrix = 1;
    constexpr int num_matrix_to_transpose = 1;
    LaunchAddBiasTranspose(stream, num_matrix_to_transpose, format, device_prop.maxThreadsPerBlock,
                           batch_size, seq_len, num_heads_, head_size,
                           reinterpret_cast<const CudaT*>(query_tensor.Data<T>()),
                           reinterpret_cast<const CudaT*>(query_bias_tensor.Data<T>()),
                           reinterpret_cast<CudaT*>(workspace.get()),
                           false, head_size, reinterpret_cast<CudaT*>(static_cast<CudaT*>(nullptr)), total_maxtrix);
  } else {
    RestorePaddingAddBiasTranspose(reinterpret_cast<const CudaT*>(query_tensor.Data<T>()),
                                   reinterpret_cast<const CudaT*>(query_bias_tensor.Data<T>()),
                                   reinterpret_cast<CudaT*>(workspace.get()),
                                   batch_size, seq_len, num_heads_, head_size,
                                   token_offset_tensor->Data<int>(),
                                   token_count, stream);
  }

  // reuse output if possible
  CudaT* gemm_output = reuse_output ? reinterpret_cast<CudaT*>(output->template MutableData<T>())
                                    : (reinterpret_cast<CudaT*>(workspace.get()) + elements_in_query);
  int ld_gemm_output = reuse_output ? seq_len : static_cast<int>(D);

  const CudaT one = ToCudaType<T>::FromFloat(1.0f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // ([b*n*s, h] * [h, D]), CUDA assumes col-major
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      D, BNS, head_size, &one,
      reinterpret_cast<const CudaT*>(weight_tensor.template Data<T>()), (int)D,
      reinterpret_cast<const CudaT*>(workspace.get()), (int)head_size,
      &zero, gemm_output, ld_gemm_output, device_prop));

  auto status = LaunchGatedRelativePositionBiasKernel<CudaT>(
      device_prop, stream,
      reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      reinterpret_cast<const CudaT*>(rel_pos_tensor.template Data<T>()),
      reinterpret_cast<const CudaT*>(gemm_output),
      reinterpret_cast<const CudaT*>(bias_tensor.template Data<T>()),
      reinterpret_cast<const CudaT*>(eco_a_tensor.template Data<T>()),
      batch_size, num_heads_, seq_len, D, ld_gemm_output);

  return status;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
