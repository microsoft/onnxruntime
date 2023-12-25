// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class DequantizeAndUnpackWeight final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit DequantizeAndUnpackWeight(const OpKernelInfo& info) : CudaKernel{info} {
    ORT_ENFORCE(info.GetAttr<int64_t>("bits", &bits_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("groupsize", &group_size_).IsOK());
    in_features_ = info.GetAttrOrDefault<int64_t>("in_features", -1);

    ORT_ENFORCE(bits_ > 1 && bits_ < 9, "bits must be in range [2, 8]");
    if (bits_ != 2 && bits_ != 4 && bits_ != 8 && in_features_ == -1) {
      ORT_THROW("in_features must be specified for bits other than 2, 4, 8");
    }
    if (in_features_ == -1) {
      const auto& node{Node()};
      const auto& input_defs = node.InputDefs();
      const NodeArg& X = *input_defs[0];
      in_features_ = X.Shape()->dim(0).dim_value() * (32 / bits_);
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  using Base = CudaKernel;
  int64_t bits_;
  int64_t group_size_;
  int64_t in_features_;
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
    uint32_t bits,
    uint32_t groupsize);
void DequantWeightNbit_g(cudaStream_t stream,
                         const int32_t* qweight_i32,
                         const void* scales_data,
                         const int32_t* zeros_data,
                         const int32_t* g_dix,
                         void* b_fp16,
                         uint32_t mat_k, uint32_t mat_n, int bits,
                         int groupsize);

Status DequantizeAndUnpackWeight::ComputeInternal(OpKernelContext* ctx) const {
  const auto* qweight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);
  const auto* g_idx = ctx->Input<Tensor>(3);

  auto output_shape = qweight->Shape();
  output_shape[0] = in_features_;

  auto* output = ctx->Output(0, output_shape);
  if (g_idx && g_idx->Shape().Size() > 1) {
    DequantWeightNbit_g(Stream(ctx), qweight->Data<int32_t>(),
                        input_scale->Data<MLFloat16>(),
                        input_zeros->Data<int32_t>(),
                        g_idx->Data<int32_t>(),
                        output->MutableData<MLFloat16>(),
                        in_features_, output_shape[1], bits_, group_size_);
  }else{
    DequantWeightNbit(Stream(ctx), qweight->Data<int32_t>(),
                      input_scale->Data<MLFloat16>(),
                      input_zeros->Data<int32_t>(),
                      output->MutableData<MLFloat16>(),
                      in_features_, output_shape[1], bits_, group_size_);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
