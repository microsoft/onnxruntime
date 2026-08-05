// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/compressed_sparse_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
class CompressedSparseAttention final : public CudaKernel {
 public:
  explicit CompressedSparseAttention(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate_).IsOK() && compress_rate_ > 0);
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ % 2 == 0);
    epsilon_ = info.GetAttrOrDefault<float>("rms_norm_epsilon", 1e-6f);
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    for (int index = 0; index < 13; ++index) ORT_RETURN_IF_NOT(context->Input<Tensor>(index), "all CSA inputs are required.");
    CompressorState<T> state;
    ORT_RETURN_IF_ERROR(RunCompressor<T>(
        *this, context, *context->Input<Tensor>(0), *context->Input<Tensor>(2), *context->Input<Tensor>(3),
        *context->Input<Tensor>(4), *context->Input<Tensor>(5), *context->Input<Tensor>(6),
        *context->Input<Tensor>(7), *context->Input<Tensor>(8), *context->Input<Tensor>(9),
        *context->Input<Tensor>(10), context->Input<Tensor>(11), context->Input<Tensor>(12),
        compress_rate_, rotary_dim_, epsilon_, 0, 1, 2, 4, 5, state));
    Tensor* present_entries = context->Output(3, context->Output<Tensor>(0)->Shape());
    return CUDA_CALL(cudaMemcpyAsync(present_entries->MutableData<T>(), state.entries,
                                     static_cast<size_t>(present_entries->Shape().Size()) * sizeof(T),
                                     cudaMemcpyDeviceToDevice, Stream(context)));
  }
 private:
  int64_t compress_rate_{};
  int64_t rotary_dim_{};
  float epsilon_{};
};

}  // namespace

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CompressedSparseAttention, kMSDomain, 1, T, kCudaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      CompressedSparseAttention<T>);

REGISTER_KERNEL(MLFloat16)
REGISTER_KERNEL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
