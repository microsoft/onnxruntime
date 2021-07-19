// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_bias_gelu.h"

#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/math/gemm_matmul_common.h"
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

template <typename T>
void ComputeMatMul(const int64_t sequence_size,
                   concurrency::ThreadPool* thread_pool,
                   MatMulComputeHelper& helper,
                   bool trans_a,
                   bool trans_b,
                   bool is_packed,
                   const T* a_data,
                   const T* b_data,
                   const BufferUniquePtr& packed_b,
                   T* output_data) {
  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M()) / sequence_size;
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = static_cast<int>(trans_a ? M : K);
  const size_t ldb = static_cast<int>(trans_b ? K : N);

  // TODO(kreeger): Move this stuff down into the ::Compute() block.
  MLAS_SGEMM_DATA_PARAMS data;
  data.BIsPacked = is_packed;
  data.A = a_data + helper.LeftOffsets()[0];
  data.lda = lda;
  data.B = is_packed ? static_cast<float*>(packed_b.get())
                     : b_data + helper.RightOffsets()[0];
  data.ldb = ldb;
  data.C = output_data + helper.OutputOffsets()[0];
  data.ldc = N;
  //data.alpha = alpha_attr_;  // TODO - fix this!
  data.beta = 0.0f;

  // TODO - can't actually call nested stuff here!
  // TODO( Handle CblasTrans from matmul impl!
  MlasGemm(CblasNoTrans,
           CblasNoTrans,
           M,
           N,
           K,
           data,
           thread_pool);
}

// TODO - this is really a vector dot product...
template <typename T>
void ComputeMatMulSlow(const int64_t a,
                       const int64_t b,
                       const T* a_data,
                       const T* b_data,
                       T* output_data) {
  for (int64_t i = 0; i < b; ++i) {
    T sum = 0;
    for (int64_t j = 0; j < a; ++j) {
      sum += a_data[j] * b_data[i + (b * j)];
    }
    output_data[i] = sum;
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

// TODO - handle hybrid types or stick to float?
template <typename T>
Status EmbedLayerNormBiasGelu<T>::PrePack(
    const Tensor& tensor,
    int input_idx,
    AllocatorPtr alloc,
    /*out*/ bool& is_packed,
    /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  // Only pack Matrix 'B' of both matmuls:
  //if (input_idx == 5) {
  //  size_t packed_matmul_1_b_size;
  //  is_packed = GemmPackBFp32(alloc,
  //                            tensor,
  //                            false /* matmul_1_trans_b_attr_*/,
  //                            matmul_1_packed_b_,
  //                            packed_matmul_1_b_size,
  //                            matmul_1_b_shape_);
  //  if (is_packed && (prepacked_weights != nullptr)) {
  //    // Handle shared pre-packed weights:
  //    prepacked_weights->buffers_.push_back(std::move(matmul_1_packed_b_));
  //    prepacked_weights->buffer_sizes_.push_back(packed_matmul_1_b_size);
  //  }
  //} else if (input_idx == 7) {
  //  size_t packed_matmul_2_b_size;
  //  is_packed = GemmPackBFp32(alloc,
  //                            tensor,
  //                            false /* matmul_1_trans_b_attr_*/,
  //                            matmul_2_packed_b_,
  //                            packed_matmul_2_b_size,
  //                            matmul_2_b_shape_);
  //  if (is_packed && (prepacked_weights != nullptr)) {
  //    // Handle shared pre-packed weights:
  //    prepacked_weights->buffers_.push_back(std::move(matmul_2_packed_b_));
  //    prepacked_weights->buffer_sizes_.push_back(packed_matmul_2_b_size);
  //  }
  //}

  return Status::OK();
}

// TODO - handle hybrid types or stick to float?
template <typename T>
Status EmbedLayerNormBiasGelu<T>::UseSharedPrePackedBuffers(
    std::vector<BufferUniquePtr>& prepacked_buffers,
    int input_idx,
    /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (input_idx == 5) {
    used_shared_buffers = true;
    matmul_1_packed_b_ = std::move(prepacked_buffers[0]);
  } else if (input_idx == 7) {
    used_shared_buffers = true;
    matmul_2_packed_b_ = std::move(prepacked_buffers[0]);
  }

  return Status::OK();
}

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
        //ComputeMatMul(sequence_length,
        //              context->GetOperatorThreadPool(),
        //              matmul_1_helper,
        //              /*trans_a=*/false,
        //              /*trans_b=*/false,
        //              /*is_packed=*/matmul_1_b == nullptr,
        //              skip_layer_norm_output_data + (task_idx * hidden_size),
        //              matmul_1_b_data,
        //              matmul_1_packed_b_,
        //              matmul_1_output + offset);
        ComputeMatMulSlow(hidden_size,
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
        //ComputeMatMul(sequence_length,
        //              context->GetOperatorThreadPool(),
        //              matmul_2_helper,
        //              /*trans_a=*/false,
        //              /*trans_b=*/false,
        //              /*is_packed=*/matmul_2_b == nullptr,
        //              bias_gelu_output + offset,
        //              matmul_2_b_data,
        //              matmul_2_packed_b_,
        //              output_data + (task_idx * hidden_size));
        ComputeMatMulSlow(bias_size,
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
