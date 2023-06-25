// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class DequantizeAndUnpackWeight final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit DequantizeAndUnpackWeight(const OpKernelInfo& info) : CudaKernel{info} {
    ORT_ENFORCE(info.GetAttr<int64_t>("bits", &bits_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("groupsize", &group_size_).IsOK());
    ORT_ENFORCE(bits_ == 8 || bits_ == 4, "bits must be 8 or 4");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  using Base = CudaKernel;
  int64_t bits_;
  int64_t group_size_;
};

ONNX_OPERATOR_KERNEL_EX(
    DequantizeAndUnpackWeight,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t>())
        .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16>()),
    DequantizeAndUnpackWeight);

void DequantWeightNbit(
    cudaStream_t stream,
    const int32_t* qweight_i32,
    const void* scales_data,
    const int32_t* zeros_data,
    void* weight_out,
    uint32_t MATRIX_K,
    uint32_t MATRIX_N,
    uint32_t groupsize);

Status DequantizeAndUnpackWeight::ComputeInternal(OpKernelContext* ctx) const {
  const auto* qweight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);

  auto qweight_shape = qweight->Shape();
  qweight_shape[0] *= 32 / bits_;

  auto* output = ctx->Output(0, qweight_shape);
  DequantWeightNbit(Stream(ctx), qweight->Data<int32_t>(),
                    input_scale->Data<MLFloat16>(),
                    input_zeros->Data<int32_t>(),
                    output->MutableData<MLFloat16>(),
                    qweight->Shape()[0], qweight_shape[1], group_size_);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
