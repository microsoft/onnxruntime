// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_bias_gelu.h"

#include "contrib_ops/cpu/skip_layer_norm.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/gemm_matmul_common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {

namespace {

#pragma warning(disable : 4189 4100)

template <typename T>
Status CheckInputs(const Tensor* input,
                   const Tensor* skip,
                   const Tensor* gamma,
                   const Tensor* beta,
                   const Tensor* bias,
                   const Tensor* matmul_1_b,
                   const Tensor* bias_gelu_bias,
                   const Tensor* matmul_2_b) {
  // First set of inputs should match existing requirements for SkipLayerNorm:
  ORT_RETURN_IF_ERROR(
      SkipLayerNorm<T>::CheckInputs(input, skip, gamma, beta, bias));

  const auto& input_dims = input->Shape().GetDims();

  // TODO(kreeger): handle packed weights (check nullptr on packed tensors).

  // Ensure that MatMul #1, BiasGelu, and MatMul #2 match the dimension
  // requirements of the SkipLayerNorm.
  const auto matmul_1_b_shape = matmul_1_b->Shape().GetDims();
  if (matmul_1_b_shape.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMul #1 is not a 2 dimensional tensor");
  }
  if (matmul_1_b_shape[0] != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMul #1 first dim: ", matmul_1_b_shape[0],
                           " does not match hidden size: ", input_dims[2]);
  }

  const auto& bias_gelu_bias_shape = bias_gelu_bias->Shape().GetDims();
  if (bias_gelu_bias_shape.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Bias Gelu bias is not a 1 dimensional tensor");
  }
  if (bias_gelu_bias_shape[0] != matmul_1_b_shape[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Bias Gelu second dim: ", bias_gelu_bias_shape[0],
                           " does not match bias size: ", matmul_1_b_shape[1]);
  }

  const auto matmul_2_b_shape = matmul_2_b->Shape().GetDims();
  if (matmul_2_b_shape.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMul #2 is not a 2 dimensional tensor");
  }
  if (matmul_2_b_shape[0] != matmul_1_b_shape[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMul #2 first dim: ", matmul_2_b_shape[0],
                           " does not match bias size: ", matmul_1_b_shape[1]);
  }
  if (matmul_2_b_shape[1] != input_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MatMul #2 first dim: ", matmul_2_b_shape[1],
                           " does not match hidden size: ", input_dims[2]);
  }

  return Status::OK();
}

template <typename T>
void ComputeSkipLayerNorm(int64_t hidden_size,
                          float epsilon,
                          const T* input_data,
                          const T* skip_data,
                          const T* gamma_data,
                          const T* beta_data,
                          const T* bias_data,
                          T* output_data) {
  T mean = 0;
  T mean_square = 0;

  for (int64_t i = 0; i < hidden_size; ++i) {
    T value = input_data[i] + skip_data[i];
    if (bias_data != nullptr) {
      value += bias_data[i];
    }

    output_data[i] = value;
    mean += value;
    mean_square += value * value;
  }

  mean = mean / hidden_size;
  mean_square = sqrt(mean_square / hidden_size - mean * mean + epsilon);

  for (int64_t i = 0; i < hidden_size; ++i) {
    if (beta_data == nullptr) {
      output_data[i] =
          (output_data[i] - mean) / mean_square * gamma_data[i];
    } else {
      output_data[i] =
          (output_data[i] - mean) / mean_square * gamma_data[i] + beta_data[i];
    }
  }
}

// TODO(kreeger): replace this with the MlasVectorDotProduct call.
template <typename T>
void VectorDotProductSlow(const int64_t M,
                          const int64_t N,
                          const T* A,
                          const T* B,
                          T* C) {
  for (int64_t i = 0; i < N; ++i) {
    T sum = 0;
    for (int64_t j = 0; j < M; ++j) {
      sum += A[j] * B[i + (N * j)];
    }
    C[i] = sum;
  }
}

template <typename T>
void ComputeBiasGelu(const int64_t bias_size,
                     const T* input_data,
                     const T* bias_data,
                     T* output_data) {
  // TODO(kreeger): Handle bias_data == nullptr (FastGelu?)
  // TODO(kreeger): Consider allocating buffer if bias_size is too big:
  std::vector<T> temp;
  temp.resize(bias_size);

  for (int64_t i = 0; i < bias_size; ++i) {
    T cur_value = input_data[i] + bias_data[i];
    output_data[i] = cur_value * static_cast<T>(M_SQRT1_2);
    temp[i] = cur_value * 0.5f;
  }

  MlasComputeErf(output_data, output_data, bias_size);

  for (int64_t i = 0; i < bias_size; ++i) {
    output_data[i] = temp[i] * (output_data[i] + 1.0f);
  }
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormBiasGelu,                                     \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNormBiasGelu<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
EmbedLayerNormBiasGelu<T>::EmbedLayerNormBiasGelu(
    const OpKernelInfo& op_kernel_info)
    : EmbedLayerNormBase(op_kernel_info) {}

template <typename T>
Status EmbedLayerNormBiasGelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* skip = context->Input<Tensor>(1);
  const Tensor* gamma = context->Input<Tensor>(2);
  const Tensor* beta = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* matmul_1_b = matmul_1_packed_b_ ? nullptr : context->Input<Tensor>(5);
  const Tensor* bias_gelu_bias = context->Input<Tensor>(6);
  const Tensor* matmul_2_b = matmul_1_packed_b_ ? nullptr : context->Input<Tensor>(7);

  ORT_RETURN_IF_ERROR(CheckInputs<T>(input,
                                     skip,
                                     gamma,
                                     beta,
                                     bias,
                                     matmul_1_b,
                                     bias_gelu_bias,
                                     matmul_2_b));

  const TensorShape& output_shape = input->Shape();

  Tensor* skip_layer_norm_output = context->Output(0, output_shape);
  Tensor* output = context->Output(1, output_shape);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const auto& input_dims = input->Shape().GetDims();

  const int64_t batch_size = input_dims[0];
  const int64_t sequence_length = input_dims[1];
  const int64_t hidden_size = input_dims[2];
  const int64_t bias_size = bias_gelu_bias->Shape().GetDims()[0];

  //========================== SLN =============================================
  const T* input_data = input->Data<T>();
  const T* skip_data = skip->Data<T>();
  const T* gamma_data = gamma->Data<T>();
  // TODO(kreeger): add unit tests for this:
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* skip_layer_norm_output_data = skip_layer_norm_output->MutableData<T>();
  T* output_data = output->MutableData<T>();

  //========================== MatMul #1 =======================================
  // Determine shape and size for matmul #1:
  const auto matmul_1_b_shape =
      matmul_1_b != nullptr ? matmul_1_b->Shape() : matmul_1_b_shape_;

  MatMulComputeHelper matmul_1_helper;
  ORT_RETURN_IF_ERROR(matmul_1_helper.Compute(output_shape,
                                              matmul_1_b_shape,
                                              /*transa=*/false,
                                              /*transb=*/false));
  const T* matmul_1_b_data =
      matmul_1_b != nullptr ? matmul_1_b->Data<T>() : nullptr;

  // Scratch buffer for MatMul #1 output:
  auto matmul_1_output_data =
      alloc->Alloc(SafeInt<size_t>(matmul_1_helper.OutputShape().Size() * sizeof(T)));
  BufferUniquePtr matmul_1_output_buffer(matmul_1_output_data,
                                         BufferDeleter(alloc));
  T* matmul_1_output = reinterpret_cast<T*>(matmul_1_output_data);

  //========================== BiasGelu =1======================================

  // BiasGelu shape is the output of matmul #1:
  const T* bias_gelu_bias_data = bias_gelu_bias->Data<T>();

  auto bias_gelu_shape = matmul_1_helper.OutputShape();
  auto bias_gelu_output_data =
      alloc->Alloc(SafeInt<size_t>(bias_gelu_shape.Size() * sizeof(T)));
  BufferUniquePtr bias_gelu_output_buffer(bias_gelu_output_data,
                                          BufferDeleter(alloc));
  T* bias_gelu_output = reinterpret_cast<T*>(bias_gelu_output_data);

  //========================== MatMul #2 =======================================

  const auto matmul_2_b_shape =
      matmul_2_b != nullptr ? matmul_2_b->Shape() : matmul_2_b_shape_;

  // Determine shape and size for matmul #2:
  MatMulComputeHelper matmul_2_helper;
  ORT_RETURN_IF_ERROR(matmul_2_helper.Compute(bias_gelu_shape,
                                              matmul_2_b_shape,
                                              /*transa=*/false,
                                              /*transb=*/false));
  const T* matmul_2_b_data =
      matmul_2_b != nullptr ? matmul_2_b->Data<T>() : nullptr;

  //----------------------------------------------------------------------------


  //
  // TODO(kreeger): LEFT OFF RIGHT HERE:
  // 1.) Experiment with task count or give up batches for f32
  // 2.) Move to fused implementation with shared code for SLN, MatMul, and BiasGelu.
  // 3.) Determine if a fusing can be done for quantization only.
  //

  // NOTE: this was taken from what is in SLN.
  int64_t task_count = batch_size * sequence_length;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        // TODO(kreeger): Add some ASCII art to help with the breakup of this
        //                work as it filters through the graph.

        // First, calculate the SkipLayerNorm output for the current task
        ComputeSkipLayerNorm(hidden_size,
                             epsilon(),
                             input_data + (task_idx * hidden_size),
                             skip_data + (task_idx * hidden_size),
                             gamma_data,
                             beta_data,
                             bias_data,
                             skip_layer_norm_output_data + (task_idx * hidden_size));

        // Now perform MatMul on the 1 row that was calculated in the call to
        // ComputeSkipLayerNorm():
        size_t offset = task_idx * bias_size;
        VectorDotProductSlow(hidden_size,
                             bias_size,
                             skip_layer_norm_output_data + (task_idx * hidden_size),
                             matmul_1_b_data,
                             matmul_1_output + offset);

        // Use the row calculated in the MatMul #1 and pass through for BiasGelu
        // calculation:
        ComputeBiasGelu(bias_size,
                        matmul_1_output + offset,
                        bias_gelu_bias_data,
                        bias_gelu_output + offset);

        // Finally, perform one more MatMul on the row calculated at the start
        // of this batch:
        VectorDotProductSlow(bias_size,
                             hidden_size,
                             bias_gelu_output + offset,
                             matmul_2_b_data,
                             output_data + (task_idx * hidden_size));
      },
      0);

  return Status::OK();
}
#pragma warning(default : 4189 4100)

}  // namespace contrib
}  // namespace onnxruntime
