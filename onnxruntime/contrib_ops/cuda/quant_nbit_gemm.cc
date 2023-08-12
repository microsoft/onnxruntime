// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_kernel.h"

#include "quant_nbit_gemm.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class QuantNbitsGemm final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit QuantNbitsGemm(const OpKernelInfo& info) : CudaKernel{info} {
    // ORT_ENFORCE(info.GetAttr("out_features", &outfeatures_).IsOK());
    ORT_ENFORCE(info.GetAttr("in_features", &in_features_).IsOK());
    bits_ = info.GetAttrOrDefault<int64_t>("bits", 3);
    groupsize_ = info.GetAttrOrDefault<int64_t>("groupsize", 128);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  using Base = CudaKernel;
  using CublasHandle = cublasHandle_t;

  template <typename T>
  struct ComputeImpl;

  // int64_t outfeatures_;
  int64_t in_features_;
  int64_t bits_;
  int64_t groupsize_;
};

ONNX_OPERATOR_KERNEL_EX(
    QuantNbitsGemm,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>()),
    QuantNbitsGemm);

void DequantWeightNbit(
    cudaStream_t stream,
    const int32_t* qweight_i32,
    const void* scales_data,
    const int32_t* zeros_data,
    void* weight_out,
    uint32_t MATRIX_K,
    uint32_t MATRIX_N,
    uint32_t bits,
    uint32_t groupsize);

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

Status QuantNbitsGemm::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_x = ctx->Input<Tensor>(0);
  const auto* input_qweight = ctx->Input<Tensor>(1);
  const auto* input_scale = ctx->Input<Tensor>(2);
  const auto* input_zeros = ctx->Input<Tensor>(3);
  ////const auto* input_bias = ctx->Input<Tensor>(4);
  ////const auto* input_gidx = ctx->Input<Tensor>(5);
  const auto& input_shape = input_x->Shape();
  const auto& weight_shape = input_qweight->Shape();
  if (input_shape.NumDimensions() == 2 && input_shape[0] <= 4) {
    TensorShapeVector output_shape = input_shape.AsShapeVector();
    output_shape[output_shape.size() - 1] = weight_shape[1];
    auto* output = ctx->Output(0, output_shape);
    auto batch = input_shape[0] * (input_shape.NumDimensions() > 2 ? input_shape[1] : 1);
    int64_t in_features = input_shape[input_shape.NumDimensions() - 1];

    Q4bitGemv(Stream(ctx), input_x->Data<MLFloat16>(),
                     input_qweight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                     input_scale->Data<MLFloat16>(), input_zeros->Data<int32_t>(),
                     batch, in_features, weight_shape[1],
                     groupsize_);
    return Status::OK();
  } else {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

    auto fp16_weight_shape = input_qweight->Shape();
    fp16_weight_shape[0] *= 32 / bits_;

    auto temp_fp16_weight = Tensor::Create(input_scale->DataType(), fp16_weight_shape, alloc);

    DequantWeightNbit(Stream(ctx), input_qweight->Data<int32_t>(),
                      input_scale->Data<MLFloat16>(),
                      input_zeros->Data<int32_t>(),
                      temp_fp16_weight->MutableData<MLFloat16>(),
                      weight_shape[0], weight_shape[1], bits_, groupsize_);
    return Fp16GemmHelper<MLFloat16>(ctx, temp_fp16_weight.get(), GetDeviceProp(), GetCublasHandle(ctx));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
