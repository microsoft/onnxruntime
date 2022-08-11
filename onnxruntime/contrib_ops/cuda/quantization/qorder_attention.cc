// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qorder_common.h"
#include "qorder_common_impl.h"
#include "qorder_attention.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include <iostream>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define DefineQOrderedAttentionInput(a, s, id) a = id
enum InputIds {
#include "./qorder_attention_input_enum.h"
};
#undef DefineQOrderedAttentionInput

static_assert((InputIds::K_Weight == 1 + InputIds::Q_Weight) && (InputIds::V_Weight == 2 + InputIds::Q_Weight));
static_assert((InputIds::Scale_K_Weight == 1 + InputIds::Scale_Q_Weight) && (InputIds::Scale_V_Weight == 2 + InputIds::Scale_Q_Weight));
static_assert((InputIds::K_Bias == 1 + InputIds::Q_Bias) && (InputIds::V_Bias == 2 + InputIds::Q_Bias));

ONNX_OPERATOR_KERNEL_EX(
    QOrderedAttention,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", BuildKernelDefConstraints<float>())
        .TypeConstraint("G", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::ScaleInput)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_Q_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_K_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_V_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_QKT_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_QKT_Softmax)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_Values_Gemm),
    QOrderedAttention);

QOrderedAttention::QOrderedAttention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info), single_weight_shape_{0LL, 0LL} {
  unidirectional_ = info.GetAttrOrDefault("unidirectional", 0LL);
  qkv_hidden_sizes_ = info.GetAttrOrDefault("qkv_hidden_sizes", -1LL);
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight");
  order_bias_ = GetCublasLtOrderAttr(info, "order_bias");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  if (order_input_ == CUBLASLT_ORDER_ROW) {
    ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL, "Only CUBLASLT_ORDER_COL is supported for order_weight_");
    ORT_ENFORCE(order_bias_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_bias");
    ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_output");
  } else if (order_input_ == CUBLASLT_ORDER_COL32) {
    ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL4_4R2_8C || order_weight_ == CUBLASLT_ORDER_COL32_2R_4R4,
                "Only CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4 are supported for order_weight_");
    ORT_ENFORCE(order_bias_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_bias");
    ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_output");
  } else {
    ORT_ENFORCE(false, "Only CUBLASLT_ORDER_ROW or CUBLASLT_ORDER_COL32 are supported for order_input");
  }
  qkv_weight_const_count_ = scale_qkv_weight_const_count_ = qkv_bias_const_cout_ = 0;
  const_sacle_input_ = const_scale_qkv_gemm_[0] = const_scale_qkv_gemm_[1] = const_scale_qkv_gemm_[2] = 0.0f;
}

Status QOrderedAttention::PutIntoMergedWeight(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ORT_ENFORCE(tensor.Shape().NumDimensions() == 2, "QKV weight must be 2d tensors!");
  if (single_weight_shape_.Size() == 0LL) {
    single_weight_shape_ = tensor.Shape();
    if (qkv_hidden_sizes_) {
      ORT_ENFORCE(single_weight_shape_.GetDims().back() == qkv_hidden_sizes_, "qkv hidden size mot matching");
    } else {
      qkv_hidden_sizes_ = single_weight_shape_.GetDims().back();
    }
  } else {
    ORT_ENFORCE(single_weight_shape_ == tensor.Shape(), "QKV weight size should be same!");
  }
  qkv_weight_const_count_++;
  auto single_weight_bytes = single_weight_shape_.Size() * sizeof(float);
  if (!merged_qkv_weight_) {
    auto* merged_qkv_weight_data = alloc->Alloc(3 * single_weight_bytes);
    merged_qkv_weight_ = BufferUniquePtr(merged_qkv_weight_data, BufferDeleter(alloc));
  }
  float* start = (float*)(((uint8_t*)merged_qkv_weight_.get()) + single_weight_bytes * qkv_index);
  int64_t N = single_weight_shape_.Size();
  CUBLAS_RETURN_IF_ERROR(cublasScopy(CublasHandle(), gsl::narrow_cast<int>(N), tensor.Data<float>(), 1, start, 1));
  return Status::OK();
}

Status QOrderedAttention::PutIntoMergedWeightScale(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ORT_ENFORCE(qkv_hidden_sizes_ > 0 || !tensor.Shape().IsScalar(), "qkv weight columns must be know");
  if (!tensor.Shape().IsScalar()) {
    ORT_ENFORCE(tensor.Shape().NumDimensions() == 1, "qkv gemm scale must be 1d vector");
    if (qkv_hidden_sizes_ == 0) {
      qkv_hidden_sizes_ = tensor.Shape().GetDims().back();
    }
    ORT_ENFORCE(qkv_hidden_sizes_ == tensor.Shape().GetDims().back(), "qkv hidden size is not matching");
  }

  scale_qkv_weight_const_count_++;
  if (!merged_qkv_alpha_) {
    auto* merged_alpha_data = alloc->Alloc(3 * qkv_hidden_sizes_ * sizeof(float));
    merged_qkv_alpha_ = BufferUniquePtr(merged_alpha_data, BufferDeleter(alloc));
  }
  float* start = ((float*)merged_qkv_alpha_.get()) + qkv_hidden_sizes_ * qkv_index;
  int incr = (tensor.Shape().IsScalar() ? 0 : 1);
  CUBLAS_RETURN_IF_ERROR(cublasScopy(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_), tensor.Data<float>(), incr, start, 1));
  ORT_ENFORCE(const_sacle_input_ > 0.0f && const_scale_qkv_gemm_[qkv_index] > 0.0f, "input scale and repective qkv gemm scale must be constant!");
  float scale = const_sacle_input_ / const_scale_qkv_gemm_[qkv_index];
  CUBLAS_RETURN_IF_ERROR(cublasSscal(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_), &scale, start, 1));
  return Status::OK();
}

Status QOrderedAttention::PutIntoMergedBias(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ORT_ENFORCE(tensor.Shape().NumDimensions() == 1, "bias must be 1d vector");
  if (qkv_hidden_sizes_ == 0) {
    qkv_hidden_sizes_ = tensor.Shape().GetDims().back();
  }
  ORT_ENFORCE(qkv_hidden_sizes_ == tensor.Shape().GetDims().back(), "qkv hidden size is not matching");

  qkv_bias_const_cout_++;
  if (!merged_qkv_bias_) {
    auto* merged_bias_data = alloc->Alloc(3 * qkv_hidden_sizes_ * sizeof(float));
    merged_qkv_bias_ = BufferUniquePtr(merged_bias_data, BufferDeleter(alloc));
  }
  float* start = ((float*)merged_qkv_bias_.get()) + qkv_hidden_sizes_ * qkv_index;
  CUBLAS_RETURN_IF_ERROR(cublasScopy(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_), tensor.Data<float>(), 1, start, 1));
  ORT_ENFORCE(const_scale_qkv_gemm_[qkv_index] > 0.0f, "repective qkv gemm scale must be constant!");
  float scale = 1.0f / const_scale_qkv_gemm_[qkv_index];
  CUBLAS_RETURN_IF_ERROR(cublasSscal(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_), &scale, start, 1));
  return Status::OK();
}

Status QOrderedAttention::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                                  /*out*/ bool& is_packed,
                                  /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

  if (input_idx == InputIds::ScaleInput) {
    const_sacle_input_ = *tensor.Data<float>();
  }
  if (input_idx == InputIds::Scale_Q_Gemm || input_idx == InputIds::Scale_K_Gemm || input_idx == InputIds::Scale_V_Gemm) {
    const_scale_qkv_gemm_[input_idx - InputIds::Scale_Q_Gemm] = *tensor.Data<float>();
  }

  if (input_idx == InputIds::Q_Weight || input_idx == InputIds::K_Weight || input_idx == InputIds::V_Weight) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedWeight(tensor, alloc, input_idx - InputIds::Q_Weight));
  }

  if (input_idx == InputIds::Scale_Q_Weight || input_idx == InputIds::Scale_Q_Weight || input_idx == InputIds::Scale_Q_Weight) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedWeightScale(tensor, alloc, input_idx - InputIds::Scale_Q_Weight));
  }

  if (input_idx == InputIds::Q_Bias || input_idx == InputIds::K_Bias || input_idx == InputIds::V_Bias) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedBias(tensor, alloc, input_idx - InputIds::Q_Bias));
  }

  return Status::OK();
}

Status QOrderedAttention::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(qkv_bias_const_cout_ == 3 && scale_qkv_weight_const_count_ == 3 && qkv_weight_const_count_ == 3,
              "qkv gemm weight and their scales, qkv gemm bias must all be constant!");
  ORT_ENFORCE(const_sacle_input_ > 0.0f, "input scale must be constant");
  ORT_ENFORCE(std::all_of(&const_scale_qkv_gemm_[0], &const_scale_qkv_gemm_[3], [](float v) -> bool { return v > 0.0f; }),
              "All qkv gemm output scale must be constant!");
  // inputs are column based
  const Tensor* input = context->Input<Tensor>(InputIds::Input);
  TensorShapeVector merged_shape = ToShapeVector(single_weight_shape_.GetDims());
  merged_shape.back() *= 3;
  TensorShape merged_weights_shape(merged_shape);
  TensorShape merged_bias_shape{qkv_hidden_sizes_ * 3};
  const Tensor* mask_index = context->Input<Tensor>(InputIds::Mask_Index);

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), merged_weights_shape, merged_bias_shape, mask_index, nullptr, nullptr, device_prop.maxThreadsPerBlock));

  const Tensor* scale_output = context->Input<Tensor>(InputIds::Scale_Values_Gemm);
  const float* scale_output_data = scale_output->template Data<float>();

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int input_hidden_size = static_cast<int>(shape[2]);
  int hidden_size = qkv_hidden_sizes_;
  int head_size = hidden_size / num_heads_;

  TensorShapeVector output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasLtHandle_t cublasLt = CublasLtHandle();
  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer_quantized = GetScratchBuffer<int8_t>(2 * m * n);  // col32 + row

  cudaStream_t stream = Stream();

  // Gemm result(M, N) = alpha * input * scale_weights * weights + scale_bias.
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      1, m, n, k,
                                      (const float*)merged_qkv_alpha_.get(), input->template Data<int8_t>(), (const int8_t*)merged_qkv_weight_.get(),
                                      (const float*)merged_qkv_bias_.get(), gemm_buffer_quantized.get(),
                                      (cublasLtOrder_t)order_weight_, CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO));

  using CudaT = ToCudaType<MLFloat16>::MappedType;
  constexpr size_t element_size = sizeof(MLFloat16);

  auto gemm_buffer = GetScratchBuffer<int8_t>(m * n * element_size);  // row, fp16

  const int8_t* batch_src = ((const int8_t*)gemm_buffer_quantized.get());
  CudaT* batch_dst = ((CudaT*)gemm_buffer.get());
  int64_t single_gemm_result_size = sequence_length * hidden_size;
  for (int b = 0; b < batch_size; b++) {
    for (int qkv = 0; qkv < 3; qkv++) {
      QOrderDequantizeToRow((cublasLtOrder_t)order_input_, stream, GetDeviceProp(), batch_src, batch_dst,
                            const_scale_qkv_gemm_[qkv], batch_size, sequence_length, n);
      batch_src += single_gemm_result_size;
      batch_dst += single_gemm_result_size;
    }
  }

  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size, batch_size, num_heads_, head_size, sequence_length, 0);
  auto temp_buffer = GetScratchBuffer<void>(workSpaceSize);
  auto output_buffer = GetScratchBuffer<int8_t>(m * n * element_size);  // row, fp16
  cublasHandle_t cublas = CublasHandle();
  if (!LaunchAttentionKernel(
          device_prop,
          stream,
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
          nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
          reinterpret_cast<CudaT*>(output_buffer.get()),
          batch_size,
          sequence_length,
          num_heads_,
          head_size,
          temp_buffer.get(),
          cublas,
          element_size,
          is_unidirectional_,
          0,
          nullptr,
          nullptr,
          nullptr)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  QOrderQuantizeRowTo((cublasLtOrder_t)order_input_, stream, GetDeviceProp(),
                      (const CudaT*)output_buffer.get(), output->MutableData<int8_t>(),
                      *(const float*)scale_output_data, batch_size, sequence_length, hidden_size);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
