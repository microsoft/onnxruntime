// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulWxA16(x could be 2-8) operator, it is basically
// matmul float16 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"
#include "matmul_nbits.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_nbits.cuh"

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
  if (Y->Shape().Size() == 0)
    return Status::OK();
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
  //const int out_features = N_;
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
      GPTQPacking::DequantWeightNbitGidx(Stream(ctx), input_x->Data<MLFloat16>(),
                   input_qweight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                   input_scale->Data<MLFloat16>(), input_zeros->Data<int32_t>(),
                   input_gidx->Data<int32_t>(), shapes);
    } else {
      GPTQPacking::TryMatMul4Bits(Stream(ctx), input_x->Data<MLFloat16>(),
                  input_qweight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                  input_scale->Data<MLFloat16>(), input_zeros->Data<int32_t>(),
                  num_batches, in_features, weight_shape[1],
                  groupsize);
    }
    return Status::OK();
  } else {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

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
                                   input_zeros->Data<int32_t>(),
                                   temp_fp16_weight->MutableData<MLFloat16>(),
                                   in_features, weight_shape[1], nbits_, groupsize);
    }
    return Fp16GemmHelper<MLFloat16>(ctx, temp_fp16_weight.get(), GetDeviceProp(), GetCublasHandle(ctx));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
