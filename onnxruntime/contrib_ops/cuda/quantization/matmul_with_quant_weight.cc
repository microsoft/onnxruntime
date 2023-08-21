// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulFp32Q4 operator, it is basically
// matmul float32 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_with_quant_weight.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
class MatMulWithQuantWeight final : public CudaKernel {
 public:
  MatMulWithQuantWeight(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
};

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = trans_batch_a ? 0 : left_num_dims - 2;
  size_t right_leading_axis = trans_batch_b ? 0 : right_num_dims - 2;
  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  if (trans_batch_a) {
    left_p = left_p * left_shape[left_num_dims - 2] / left_shape[0];
  }
  int64_t left_k = transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape[right_num_dims - 2] / right_shape[0];
    }
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_leading_axis];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_leading_axis];
  int64_t m = transb ? right_shape[right_leading_axis] : right_shape[right_num_dims - 1];
  stride_A = n * left_k / (trans_batch_a ? left_shape[0] : 1);
  stride_B = right_num_dims == 2 ? 0 : right_k * m / (trans_batch_b ? right_shape[0] : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

template <typename T>
Status MatMulWithQuantWeight<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  ORT_ENFORCE(nbits_ == 4, "only 4 bits is supported now");
  ORT_ENFORCE(block_size_ == 32, "only block size 32 is supported now");

  typedef typename ToCudaType<T>::MappedType CudaT;

  IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(K_ * N_, ctx->GetComputeStream());
  auto* b_data = b_data_ptr.get();
  ORT_RETURN_IF_ERROR(Dequantize4Bits(reinterpret_cast<CudaT*>(b_data),
                                      blob_data,
                                      reinterpret_cast<const CudaT*>(scales_data),
                                      zero_points_data,
                                      SafeInt<int>(K_),
                                      SafeInt<int>(N_),
                                      SafeInt<int>(block_size_),
                                      static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
  cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  T* b_data_cpu = new T[K_ * N_];
  cudaMemcpy(b_data_cpu, b_data, K_ * N_ * sizeof(T), cudaMemcpyDeviceToHost);

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool transa = false;
  bool transb = true;

  delete[] b_data_cpu;

  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.f);

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = GetDeviceProp();

  if (helper.OutputOffsets().size() == 1) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(ctx),
        transB,
        transA,
        static_cast<int>(helper.N()),
        static_cast<int>(helper.M()),
        static_cast<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(b_data),
        ldb,
        reinterpret_cast<const CudaT*>(a_data),
        lda,
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        ldc,
        device_prop));
  } else if (CanUseStridedBatchedGemm(a->Shape(), b_shape,
                                      transa, transb, false, false, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(GetCublasHandle(ctx),
                                                          transB,
                                                          transA,
                                                          static_cast<int>(helper.N()),
                                                          static_cast<int>(helper.M()),
                                                          static_cast<int>(helper.K()),
                                                          &alpha,
                                                          reinterpret_cast<const CudaT*>(b_data),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const CudaT*>(a_data),
                                                          lda,
                                                          stride_A,
                                                          &zero,
                                                          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count),
                                                          device_prop));
  }
  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulWithQuantWeight,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulWithQuantWeight<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulWithQuantWeight,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulWithQuantWeight<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime