// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/heavily_compressed_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {


template <typename T>
class HeavilyCompressedAttention final : public CudaKernel {
 public:
  explicit HeavilyCompressedAttention(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate_).IsOK() && compress_rate_ > 0);
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ % 2 == 0);
    epsilon_ = info.GetAttrOrDefault<float>("rms_norm_epsilon", 1e-6f);
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    for (int index = 0; index < 11; ++index) ORT_RETURN_IF_NOT(context->Input<Tensor>(index), "all HCA inputs are required.");
    const Tensor& hidden = *context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(context->Input<Tensor>(1)->Shape() == TensorShape({hidden.Shape()[0], hidden.Shape()[1]}),
                      "position_ids shape mismatch.");
    CompressorState<T> state;
    ORT_RETURN_IF_ERROR(RunCompressor<T>(
        *this, context, hidden, *context->Input<Tensor>(2), *context->Input<Tensor>(3),
        *context->Input<Tensor>(4), *context->Input<Tensor>(5), *context->Input<Tensor>(6),
        *context->Input<Tensor>(7), *context->Input<Tensor>(8), *context->Input<Tensor>(9),
        *context->Input<Tensor>(10), nullptr, nullptr, compress_rate_, rotary_dim_, epsilon_,
        0, 2, 3, -1, -1, state));
    const TensorShape entries_shape({hidden.Shape()[0], 1, state.entry_count, state.head_size});
    ORT_RETURN_IF_NOT(state.rank4, "HCA past_entries must have rank 4.");
    Tensor* present_entries = context->Output(4, entries_shape);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_entries->MutableData<T>(), state.entries,
                                         static_cast<size_t>(entries_shape.Size()) * sizeof(T),
                                         cudaMemcpyDeviceToDevice, Stream(context)));
    Tensor* block_bias = context->Output(1, TensorShape({hidden.Shape()[0], 1, hidden.Shape()[1], state.entry_count}));
    using CudaT = typename ToCudaType<T>::MappedType;
    return LaunchHcaBlockBiasKernel<CudaT>(
        Stream(context), reinterpret_cast<CudaT*>(block_bias->MutableData<T>()),
        context->Input<Tensor>(1)->Data<int64_t>(), narrow<int>(hidden.Shape()[0]),
        narrow<int>(hidden.Shape()[1]), state.entry_count, narrow<int>(compress_rate_),
        GetDeviceProp().maxThreadsPerBlock);
  }
 private:
  int64_t compress_rate_{};
  int64_t rotary_dim_{};
  float epsilon_{};
};

}  // namespace

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      HeavilyCompressedAttention, kMSDomain, 1, T, kCudaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      HeavilyCompressedAttention<T>);

REGISTER_KERNEL(MLFloat16)
REGISTER_KERNEL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
