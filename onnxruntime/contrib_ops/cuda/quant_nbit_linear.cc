// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "quant_nbit_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class QuantNbitsLinear final : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit QuantNbitsLinear(const OpKernelInfo& info) : CudaKernel{info} {
    //ORT_ENFORCE(info.GetAttr("outfeatures", &outfeatures_).IsOK());
    //ORT_ENFORCE(info.GetAttr("infeatures", &in_features_).IsOK());
    bits_ = info.GetAttrOrDefault<int64_t>("bits", 3);
    groupsize_ = info.GetAttrOrDefault<int64_t>("groupsize", 128);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  using Base = CudaKernel;
  using CublasHandle = cublasHandle_t;

  template <typename T>
  struct ComputeImpl;

  int64_t outfeatures_;
  int64_t in_features_;
  int64_t bits_;
  int64_t groupsize_;
};

ONNX_OPERATOR_KERNEL_EX(
    QuantNbitsLinear,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>()),
    QuantNbitsLinear);

Status QuantNbitsLinear::ComputeInternal(OpKernelContext* ctx) const {
  const auto* input_x = ctx->Input<Tensor>(0);
  const auto* input_weight = ctx->Input<Tensor>(1);
  const auto* input_scale = ctx->Input<Tensor>(2);
  const auto* input_zeros = ctx->Input<Tensor>(3);
  ////const auto* input_bias = ctx->Input<Tensor>(4);
  ////const auto* input_gidx = ctx->Input<Tensor>(5);
  const auto& input_shape = input_x->Shape();
  const auto& weight_shape = input_weight->Shape();
  TensorShapeVector output_shape = input_shape.AsShapeVector();
  output_shape[output_shape.size() - 1] = weight_shape[1];
  auto* output = ctx->Output(0, output_shape);
  auto batch = input_shape[0] * (input_shape.NumDimensions() > 2 ? input_shape[1] : 1);
  int64_t in_features = input_shape[input_shape.NumDimensions() - 1];
  //auto *inp=input_x->Data<MLFloat16>();
  //auto *outp=output->Data<MLFloat16>();
  //input_scale->Data<MLFloat16>();
  //printf("%zu,%zu\n", batch, output->Shape()[1]);
  quant4BGEMV_cuda(Stream(ctx), input_x->Data<MLFloat16>(),
                   input_weight->Data<int32_t>(), output->MutableData<MLFloat16>(),
                   input_scale->Data<MLFloat16>(), input_zeros->Data<int32_t>(),
                   batch, in_features, weight_shape[1],
                   groupsize_);
  //cudaDeviceSynchronize();
  //size_t sz = 1* weight_shape[1];
  //std::vector<int32_t> buf(sz);
  //std::vector<int32_t> buf1(sz);
  //cudaMemcpy(buf.data(), output->DataRaw(), sz * sizeof(MLFloat16), cudaMemcpyDeviceToHost);
  //cudaMemcpy(buf1.data(), input_weight->Data<int32_t>(), 8, cudaMemcpyDeviceToHost);
  //printf("%d...", buf[0]);
//
  //if (buf1[0] == 1718124390) {
  //  FILE* fp = fopen("out.bin", "wb");
  //  fwrite(buf.data(), sz * sizeof(MLFloat16), 1, fp);
  //  fclose(fp);
  //  printf("wirte finished, .... exiting\n");
  //  exit(0);
  //}

  // vecquant4matmul_cuda(Stream(ctx), input_x->Data<MLFloat16>(),
  //                      input_weight->Data<int32_t>(), output->MutableData<MLFloat16>(),
  //                      input_scale->Data<MLFloat16>(), input_zeros->Data<int32_t>(),
  //                      batch, weight_shape[0], weight_shape[1], input_zeros->Shape()[1],
  //                      groupsize_, in_features / 2);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
