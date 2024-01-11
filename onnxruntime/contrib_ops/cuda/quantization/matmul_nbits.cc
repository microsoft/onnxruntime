// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulFp32Q4 operator, it is basically
// matmul float32 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#include "matmul_nbits.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_nbits.cuh"
#include "dequantize_blockwise.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
static Status Fp16GemmHelper(OpKernelContext* ctx, const Tensor* weight, const cudaDeviceProp& device_prop,
                             const cublasHandle_t cublas_handle) {
  typedef typename ::onnxruntime::cuda::ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool transa = false;
  bool transb = false;
  bool trans_batch_a_ = false;
  bool trans_batch_b_ = false;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), weight->Shape(), transa, transb, trans_batch_a_, trans_batch_b_, false));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  const float alpha_ = 1.0f;
  const CudaT alpha = ::onnxruntime::cuda::ToCudaType<T>::FromFloat(alpha_);
  const CudaT zero = ::onnxruntime::cuda::ToCudaType<T>::FromFloat(0.0f);

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas_handle,
        transB,
        transA,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(weight->Data<T>()),
        ldb,
        reinterpret_cast<const CudaT*>(left_X->Data<T>()),
        lda,
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        ldc,
        device_prop));
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "this type of gemm is not supported");
}

template <typename T>
Status MatMulNBits<T>::ComputeInternalGPTQ(OpKernelContext* ctx) const {
  const int in_features = K_;
  // const int out_features = N_;
  const int groupsize = block_size_;
  const auto* input_x = ctx->Input<Tensor>(0);
  const auto* input_qweight = ctx->Input<Tensor>(1);
  const auto* input_scale = ctx->Input<Tensor>(2);
  const auto* input_zeros = ctx->Input<Tensor>(3);
  const auto* input_gidx = ctx->Input<Tensor>(4);
  const auto& input_shape = input_x->Shape();
  const auto& weight_shape = input_qweight->Shape();
  int64_t num_batches = input_shape.SizeToDimension(input_shape.NumDimensions() - 1);

  // huristic: if batch size is less than 8, use used matmul
  if (num_batches < 8) {
    TensorShapeVector output_shape = input_shape.AsShapeVector();
    output_shape[output_shape.size() - 1] = weight_shape[1];
    auto* output = ctx->Output(0, output_shape);
    if (input_gidx && input_gidx->Shape().Size() > 1) {
      int64_t shapes[5] = {
          num_batches,
          in_features,
          weight_shape[1],
          input_zeros->Shape()[1]};
      GPTQPacking::TryMatMul4BitsGidx(Stream(ctx), input_x->Data<MLFloat16>(),
                                      input_qweight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                                      input_scale->Data<MLFloat16>(), input_zeros->Data<int32_t>(),
                                      input_gidx->Data<int32_t>(), shapes);
    } else {
      GPTQPacking::TryMatMul4Bits(Stream(ctx), input_x->Data<MLFloat16>(),
                                  input_qweight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                                  input_scale->Data<MLFloat16>(), static_cast<const uint32_t*>(input_zeros->DataRaw()),
                                  num_batches, in_features, weight_shape[1],
                                  groupsize);
    }

  } else {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK()) {
      return status;
    }

    auto fp16_weight_shape = weight_shape;
    fp16_weight_shape[0] = in_features;

    auto temp_fp16_weight = Tensor::Create(input_scale->DataType(), fp16_weight_shape, alloc);
    if (input_gidx && input_gidx->Shape().Size() > 1) {
      GPTQPacking::DequantWeightNbitGidx(Stream(ctx), input_qweight->Data<int32_t>(),
                                         input_scale->Data<MLFloat16>(),
                                         input_zeros->Data<int32_t>(),
                                         input_gidx->Data<int32_t>(),
                                         temp_fp16_weight->MutableData<MLFloat16>(),
                                         in_features, weight_shape[1], nbits_, groupsize);
    } else {
      GPTQPacking::DequantWeightNbit(Stream(ctx), input_qweight->Data<int32_t>(),
                                     input_scale->Data<MLFloat16>(),
                                     static_cast<const uint32_t*>(input_zeros->DataRaw()),
                                     temp_fp16_weight->MutableData<MLFloat16>(),
                                     in_features, weight_shape[1], nbits_, groupsize);
    }
    return Fp16GemmHelper<MLFloat16>(ctx, temp_fp16_weight.get(), GetDeviceProp(), GetCublasHandle(ctx));
  }
  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::ComputeInternalHQQ(OpKernelContext* ctx) const {
  const int in_features = K_;
  // const int out_features = N_;
  const int groupsize = block_size_;
  const auto* input_x = ctx->Input<Tensor>(0);
  const auto* input_qweight = ctx->Input<Tensor>(1);
  const auto* input_scale = ctx->Input<Tensor>(2);
  const auto* input_zeros = ctx->Input<Tensor>(3);
  const auto& input_shape = input_x->Shape();
  const auto& weight_shape = input_qweight->Shape();
  int64_t num_batches = input_shape.SizeToDimension(input_shape.NumDimensions() - 1);
  typedef typename ToCudaType<MLFloat16>::MappedType CudaT;

  // huristic: if batch size is less than 8, use used matmul
  if (num_batches < 8) {
    TensorShapeVector output_shape = input_shape.AsShapeVector();
    output_shape[output_shape.size() - 1] = weight_shape[1];
    auto* output = ctx->Output(0, output_shape);
    GPTQPacking::TryMatMul4Bits(Stream(ctx), input_x->Data<MLFloat16>(),
                                input_qweight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                                input_scale->Data<MLFloat16>(), static_cast<const CudaT*>(input_zeros->DataRaw()),
                                num_batches, in_features, weight_shape[1],
                                groupsize);
  } else {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK()) {
      return status;
    }

    auto fp16_weight_shape = weight_shape;
    fp16_weight_shape[0] = in_features;

    auto temp_fp16_weight = Tensor::Create(input_scale->DataType(), fp16_weight_shape, alloc);

    GPTQPacking::DequantWeightNbit(Stream(ctx), input_qweight->Data<int32_t>(),
                                   input_scale->Data<MLFloat16>(),
                                   static_cast<const CudaT*>(input_zeros->DataRaw()),
                                   temp_fp16_weight->MutableData<MLFloat16>(),
                                   in_features, weight_shape[1], nbits_, groupsize);

    return Fp16GemmHelper<MLFloat16>(ctx, temp_fp16_weight.get(), GetDeviceProp(), GetCublasHandle(ctx));
  }
  return Status::OK();
}

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  if (packing_ == "gptq") {
    return this->ComputeInternalGPTQ(ctx);
  }
  if (packing_ == "hqq") {
    return this->ComputeInternalHQQ(ctx);
  }
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  typedef typename ToCudaType<T>::MappedType CudaT;

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  bool is_4bit_done = TryMatMul4Bits(
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      reinterpret_cast<const CudaT*>(a_data),
      blob_data,
      reinterpret_cast<const CudaT*>(scales_data),
      zero_points_data,
      SafeInt<int>(helper.M()),
      SafeInt<int>(helper.N()),
      SafeInt<int>(helper.K()),
      SafeInt<int>(block_size_),
      SafeInt<int>(GetDeviceProp().sharedMemPerBlock),
      static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  if (!is_4bit_done) {
    int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;
    IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(N_ * K_padded, ctx->GetComputeStream());
    auto* b_data = b_data_ptr.get();
    if (column_wise_quant_blk_) {
      // column-wise block
      ORT_RETURN_IF_ERROR(Dequantize4Bits(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          zero_points_data,
          SafeInt<int>(K_padded),
          SafeInt<int>(N_),
          SafeInt<int>(block_size_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
    } else {
      // row-wise block
      K_padded = K_;

      ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          zero_points_data,
          SafeInt<int>(block_size_),
          column_wise_quant_blk_,
          SafeInt<int>(K_),
          SafeInt<int>(N_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
    }
#if 0
  cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  T* b_data_cpu = new T[K_ * N_];
  cudaMemcpy(b_data_cpu, b_data, K_ * N_ * sizeof(T), cudaMemcpyDeviceToHost);
  delete[] b_data_cpu;
#endif

    const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
    const CudaT zero = ToCudaType<T>::FromFloat(0.f);

    if (helper.OutputOffsets().size() == 1) {
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          GetCublasHandle(ctx),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          SafeInt<int>(helper.N()),
          SafeInt<int>(helper.M()),
          SafeInt<int>(helper.K()),
          &alpha,
          reinterpret_cast<const CudaT*>(b_data),
          SafeInt<int>(K_padded),
          reinterpret_cast<const CudaT*>(a_data),
          helper.Lda(transa),
          &zero,
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          helper.Ldc(),
          GetDeviceProp()));
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int32_t>()}),
    MatMulNBits<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
