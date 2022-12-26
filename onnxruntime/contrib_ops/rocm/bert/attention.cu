// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/rocm/cu_inc/common.cuh>
#include "contrib_ops/rocm/bert/attention.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : RocmKernel(info), AttentionBase(info, true) {}


inline size_t AlignTo(size_t a, size_t b) {
  return onnxruntime::rocm::CeilDiv(a, b) * b;
}

inline size_t GetAttentionScratchSize(size_t element_size,
                               int batch_size,
                               int num_heads,
                               int sequence_length,
                               int all_sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * all_sequence_length;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* extra_add_qk = context->Input<Tensor>(5);

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  extra_add_qk,
                                  nullptr,
                                  device_prop.maxThreadsPerBlock));

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int input_hidden_size = static_cast<int>(shape[2]);

  // Note: Scenario where q_hidden_size == k_hidden_size != v_hidden_size is not supported in ROCM EP
  // bias shape (3 * hidden_size)
  const auto& bias_shape = bias->Shape();
  int hidden_size = static_cast<int>(bias_shape[0]) / 3;

  int head_size = hidden_size / num_heads_;

  TensorShapeVector output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  int past_sequence_length = 0;
  Tensor* present = GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length);

  rocblas_handle rocblas = GetRocblasHandle(context);
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(batch_size * sequence_length * 3 * hidden_size * element_size, context->GetComputeStream());

  size_t workspaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size,
                                                   sequence_length, past_sequence_length);
  auto workspace = GetScratchBuffer<void>(workspaceSize, context->GetComputeStream());

  const bool use_gemm_rcr_bias_permute = ParseTestOnlyEnvironmentVariable<bool>("ORT_ATTENTION_USE_GEMM_RCR_BIAS_PERMUTE", {"0", "1"}) == true;
  if (use_gemm_rcr_bias_permute) {
    ORT_ENFORCE((std::is_same_v<T, MLFloat16>));
    ORT_ENFORCE(batch_size == 64);
    ORT_ENFORCE(sequence_length == 512);
    ORT_ENFORCE(input_hidden_size == 768);

    ck::tensor_operation::device::DeviceBatchedContractionMultipleD_Xdl_CShuffle<
        1, 2, 3, 1,  // permute m2n3
        ck::half_t,
        ck::half_t,
        float,
        ck::half_t,

        ck::Tuple<ck::half_t>,

        ck::half_t,

        ck::tensor_operation::element_wise::PassThrough,

        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Add,

        ck::tensor_operation::device::GemmSpecialization::MNKPadding,

        ck::tensor_operation::device::TensorSpecialization::Packed,
        ck::tensor_operation::device::TensorSpecialization::Packed,
        ck::tensor_operation::device::TensorSpecialization::Default,

        1,
        256,  // block_size
        128,  // m_per_block
        128,  // n_per_block
        32,   // k_per_block
        8,    // ak1
        8,    // bk1
        32,   // m_per_xdl
        32,   // n_per_xdl
        2,    // m_xdl_per_wave
        2,    // n_xdl_per_wave

        ck::Sequence<4, 64, 1>,  // thread_cluster_length
        ck::Sequence<1, 0, 2>,   // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,   // src_access_order
        2,                       // src_vector_dim
        8,                       // src_scalar_per_vector
        8,                       // dst_scalar_per_vector

        1,  // add_extra_dim

        ck::Sequence<4, 64, 1>,  // thread_cluster_length
        ck::Sequence<1, 0, 2>,   // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,   // src_access_order
        2,                       // src_vector_dim
        8,                       // src_scalar_per_vector
        8,                       // dst_scalar_per_vector

        1,  // add_extra_dim

        1,                          // m_xdl_per_wave
        1,                          // n_xdl_per_wave
        ck::Sequence<1, 32, 1, 8>,  // m_n_block_wave_per_xdl
        8                           // scalar_per_vector
        > gemm_rcr_bias_permute_m2n3;

    ck::index_t M0 = batch_size;
    ck::index_t M1 = sequence_length;
    ck::index_t N0 = 3; // qkv
    ck::index_t N1 = num_heads_;  // num_head
    ck::index_t N2 = hidden_size /num_heads_; // dim_per_head, 64

    ck::index_t K0 = 768;

    ORT_ENFORCE(N1 == 12);
    ORT_ENFORCE(N2 == 64);

    std::vector<ck::index_t> a_ms_ks_lengths{1, M0, M1, K0};
    std::vector<ck::index_t> a_ms_ks_strides{M0 * M1 * K0, M1 * K0, K0, 1};

    std::vector<ck::index_t> b_ns_ks_lengths{1, N0, N1, N2, K0};
    std::vector<ck::index_t> b_ns_ks_strides{N0 * N1 * N2 * K0, N1 * N2 * K0, N2 * K0, K0, 1};

    std::vector<ck::index_t> d_ms_ns_lengths{1, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> d_ms_ns_strides{N0 * N1 * N2, 0, 0, N1 * N2, N2, 1};

    std::vector<ck::index_t> e_ms_ns_lengths{1, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> e_ms_ns_strides{M0 * M1 * N0 * N1 * N2,
                                             N1 * M1 * N2,
                                             N2,
                                             M0 * N1 * M1 * N2,
                                             M1 * N2,
                                             1};

    size_t bytes = GetAttentionScratchSize(element_size, batch_size, 12, sequence_length, past_sequence_length + sequence_length);

    if (gemm_rcr_bias_permute_m2n3_weight.get() == nullptr) {
      LOGS_DEFAULT(WARNING) << "repack weights to gemm_rcr_bias_permute_m2n3_weight";

      gemm_rcr_bias_permute_m2n3_weight = GetScratchBuffer<char>(weights->SizeInBytes(), context->GetComputeStream());
      std::vector<char> tmp(weights->SizeInBytes());
      HIP_RETURN_IF_ERROR(hipMemcpy(tmp.data(), weights->DataRaw(), weights->SizeInBytes(), hipMemcpyDeviceToHost));
      std::vector<char> packed(weights->SizeInBytes());

      // FIXME: dirty hack for 3 * 768 * 768 case
      int64_t features = weights->Shape()[0];
      int64_t out_features = weights->Shape()[1];
      ORT_ENFORCE(features * 3 == out_features);
      for (int qkv_index = 0; qkv_index < 3; qkv_index++) {
        for (int i = 0; i<features; i++) {
          for (int o = 0; o<features; o++) {
            ((T*)packed.data())[qkv_index * features * features + o * features + i] = ((T*)tmp.data())[i * 3 * features + qkv_index * features + o];
          }
        }
      }
      HIP_RETURN_IF_ERROR(hipMemcpy(gemm_rcr_bias_permute_m2n3_weight.get(), packed.data(), weights->SizeInBytes(), hipMemcpyHostToDevice));
    }

    auto invoker  = gemm_rcr_bias_permute_m2n3.MakeInvoker();
    auto argument = gemm_rcr_bias_permute_m2n3.MakeArgument(
        input->DataRaw(),
        gemm_rcr_bias_permute_m2n3_weight.get(),
        std::array<const void*, 1>{bias->DataRaw()},
        (T*)workspace.get() + 2 * (bytes / element_size), // write directly scratch3

        a_ms_ks_lengths,
        a_ms_ks_strides,
        b_ns_ks_lengths,
        b_ns_ks_strides,

        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},

        e_ms_ns_lengths,
        e_ms_ns_strides,

        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::Add{});

    invoker.Run(argument, StreamConfig{Stream(context), false});
  } else {
  typedef typename ToHipType<T>::MappedType HipT;
  namespace blas = rocm::tunable::blas;

  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  ORT_RETURN_IF_ERROR(blas::column_major::Gemm(
      IsTunableOpEnabled(), Stream(context), rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      n, m, 1,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(bias->Data<T>()), n,
      GetConstOnes<HipT>(m, Stream(context)), 1,
      /*beta=*/0.0f,
      reinterpret_cast<HipT*>(gemm_buffer.get()), n));

  // result(N, M) = 1 * weights x input + 1 x B.
  ORT_RETURN_IF_ERROR(blas::column_major::Gemm(
      IsTunableOpEnabled(), Stream(context), rocblas,
      blas::BlasOp::NonTrans, blas::BlasOp::NonTrans,
      n, m, k,
      /*alpha=*/1.0f,
      reinterpret_cast<const HipT*>(weights->Data<T>()), n,
      reinterpret_cast<const HipT*>(input->Data<T>()), k,
      /*beta=*/1.0f,
      reinterpret_cast<HipT*>(gemm_buffer.get()), n));
  }

  return LaunchAttentionKernel(
      device_prop,
      IsTunableOpEnabled(),
      Stream(context),
      rocblas,
      element_size,
      batch_size,
      sequence_length,
      num_heads_,
      head_size,
      past_sequence_length,
      is_unidirectional_,
      reinterpret_cast<const void*>(gemm_buffer.get()),
      nullptr == mask_index ? nullptr : mask_index->Data<int>(),
      nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
      mask_filter_value_,
      nullptr == past ? nullptr : past->Data<T>(),
      nullptr == extra_add_qk ? nullptr : extra_add_qk->Data<T>(),
      workspace.get(),
      output->MutableData<T>(),
      nullptr == present ? nullptr : present->MutableData<T>(),
      use_gemm_rcr_bias_permute
      );
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
