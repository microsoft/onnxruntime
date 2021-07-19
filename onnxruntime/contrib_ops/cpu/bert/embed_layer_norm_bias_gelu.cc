// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_bias_gelu.h"

#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {

namespace {

#pragma warning(disable : 4189 4100)
Status CheckInputs(const Tensor* input,
                   const Tensor* skip,
                   const Tensor* gamma,
                   const Tensor* beta,
                   const Tensor* bias,
                   const Tensor* matmul_1_b,
                   const Tensor* bias_gelu_bias,
                   const Tensor* matmul_2_b,
                   const Tensor* output) {
  //
  // TODO(kreeger): write me
  //
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

// TODO(kreeger): This is mostly a vector dot product vs. a full matmul.
// TODO(kreeger): change the var names of hidden_size and bias_size since those
//                vary based on the matmul that is getting run.
template <typename T>
void ComputeMatMul(const int64_t hidden_size,
                   const int64_t bias_size,
                   const T* a_data,
                   const T* b_data,
                   T* output_data) {
  // TODO - check inputs needs to make sure the dimensions are safe here.
  // TODO(kreeger): This is ungodly slow - need to throw this into mlas and pack?
  //for (int64_t i = 0; i < bias_size; ++i) {
  //  T sum = 0;
  //  for (int64_t j = 0; j < hidden_size; ++j) {
  //    sum += a_data[j] * b_data[i + (bias_size * j)];
  //  }
  //  output_data[i] = sum;
  //}
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
    temp[i] = cur_value * 0.5f;  // TODO(kreeger): const here.
  }

  MlasComputeErf(output_data, output_data, bias_size);

  for (int64_t i = 0; i < bias_size; ++i) {
    output_data[i] = temp[i] * (output_data[i] + 1.0f);  // TODO(kreeger): const here.
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
  const Tensor* matmul_1_b = context->Input<Tensor>(5);
  const Tensor* bias_gelu_bias = context->Input<Tensor>(6);
  const Tensor* matmul_2_b = context->Input<Tensor>(7);

  const TensorShape& output_shape = input->Shape();

  Tensor* skip_layer_norm_output = context->Output(0, output_shape);
  Tensor* output = context->Output(1, output_shape);

  ORT_RETURN_IF_ERROR(CheckInputs(input,
                                  skip,
                                  gamma,
                                  beta,
                                  bias,
                                  matmul_1_b,
                                  bias_gelu_bias,
                                  matmul_2_b,
                                  output));

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const auto& input_dims = input->Shape().GetDims();

  const int64_t batch_size = input_dims[0];
  const int64_t sequence_length = input_dims[1];
  const int64_t hidden_size = input_dims[2];
  const int64_t bias_size = bias_gelu_bias->Shape().GetDims()[0];

  const T* input_data = input->Data<T>();
  const T* skip_data = skip->Data<T>();
  const T* gamma_data = gamma->Data<T>();
  // TODO(kreeger): add unit tests for this:
  const T* beta_data = beta == nullptr ? nullptr : beta->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();

  T* skip_layer_norm_output_data = skip_layer_norm_output->MutableData<T>();
  T* output_data = output->MutableData<T>();

  // Determine shape and size for matmul #1:
  MatMulComputeHelper helper_1;
  ORT_RETURN_IF_ERROR(helper_1.Compute(output_shape,
                                       matmul_1_b->Shape(),
                                       /*transa=*/false,
                                       /*transb=*/false));
  const T* matmul_1_b_data = matmul_1_b->Data<T>();

  // Scratch buffer for matmul1 output:
  auto matmul_1_output_data =
      alloc->Alloc(SafeInt<size_t>(helper_1.OutputShape().Size() * sizeof(T)));
  BufferUniquePtr matmul_1_output_buffer(matmul_1_output_data,
                                         BufferDeleter(alloc));
  T* matmul_1_output = reinterpret_cast<T*>(matmul_1_output_data);

  // BiasGelu shape is the output of matmul #1:
  const T* bias_gelu_bias_data = bias_gelu_bias->Data<T>();

  auto bias_gelu_shape = helper_1.OutputShape();
  auto bias_gelu_output_data =
      alloc->Alloc(SafeInt<size_t>(bias_gelu_shape.Size() * sizeof(T)));
  BufferUniquePtr bias_gelu_output_buffer(bias_gelu_output_data,
                                          BufferDeleter(alloc));
  T* bias_gelu_output = reinterpret_cast<T*>(bias_gelu_output_data);

  // Determine shape and size for matmul #2:
  MatMulComputeHelper helper_2;
  ORT_RETURN_IF_ERROR(helper_2.Compute(bias_gelu_shape,
                                       matmul_2_b->Shape(),
                                       /*transa=*/false,
                                       /*transb=*/false));
  const T* matmul_2_b_data = matmul_2_b->Data<T>();

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
        // TODO(kreeger): rename this var:
        size_t offset = task_idx * bias_size;
        ComputeMatMul(hidden_size,
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
        ComputeMatMul(bias_size,
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
