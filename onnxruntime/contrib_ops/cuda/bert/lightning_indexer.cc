// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/lightning_indexer.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
class LightningIndexer final : public CudaKernel {
 public:
  explicit LightningIndexer(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate_).IsOK() && compress_rate_ > 0);
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads_).IsOK() && num_heads_ > 0);
    ORT_ENFORCE(info.GetAttr("head_size", &head_size_).IsOK() && head_size_ > 0);
    ORT_ENFORCE(info.GetAttr("index_topk", &index_topk_).IsOK() && index_topk_ > 0);
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ % 2 == 0);
    epsilon_ = info.GetAttrOrDefault<float>("rms_norm_epsilon", 1e-6f);
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    for (int index = 0; index < 16; ++index) ORT_RETURN_IF_NOT(context->Input<Tensor>(index), "all indexer inputs are required.");
    const Tensor& hidden = *context->Input<Tensor>(0);
    const Tensor& q_residual = *context->Input<Tensor>(1);
    const int64_t batch = hidden.Shape()[0];
    const int64_t sequence = hidden.Shape()[1];
    const int64_t hidden_size = hidden.Shape()[2];
    ORT_RETURN_IF_NOT(q_residual.Shape().NumDimensions() == 3 && q_residual.Shape()[0] == batch &&
                          q_residual.Shape()[1] == sequence &&
                          context->Input<Tensor>(9)->Shape() == TensorShape({q_residual.Shape()[2], num_heads_ * head_size_}) &&
                          context->Input<Tensor>(10)->Shape() == TensorShape({hidden_size, num_heads_}),
                      "indexer query projection shapes mismatch.");
    CompressorState<T> state;
    ORT_RETURN_IF_ERROR(RunCompressor<T>(
        *this, context, hidden, *context->Input<Tensor>(3), *context->Input<Tensor>(4),
        *context->Input<Tensor>(5), *context->Input<Tensor>(6), *context->Input<Tensor>(7),
        *context->Input<Tensor>(8), *context->Input<Tensor>(11), *context->Input<Tensor>(12),
        *context->Input<Tensor>(13), context->Input<Tensor>(14), context->Input<Tensor>(15),
        compress_rate_, rotary_dim_, epsilon_, 3, 1, 2, 4, 5, state));
    using CudaT = typename ToCudaType<T>::MappedType;
    auto queries = GetScratchBuffer<CudaT>(static_cast<size_t>(batch * sequence * num_heads_ * head_size_), Stream(context));
    auto head_weights = GetScratchBuffer<CudaT>(static_cast<size_t>(batch * sequence * num_heads_), Stream(context));
    ORT_RETURN_IF_ERROR(Project<T>(*this, context, q_residual, *context->Input<Tensor>(9),
                     narrow<int>(batch * sequence), narrow<int>(q_residual.Shape()[2]),
                     narrow<int>(num_heads_ * head_size_), queries.get()));
    ORT_RETURN_IF_ERROR(Project<T>(*this, context, hidden, *context->Input<Tensor>(10),
                     narrow<int>(batch * sequence), narrow<int>(hidden_size),
                     narrow<int>(num_heads_), head_weights.get()));
    Tensor* selected = context->Output(0, TensorShape({batch, sequence, index_topk_}));
    return LaunchLightningIndexerKernel<CudaT>(
        Stream(context), selected->MutableData<int64_t>(), queries.get(), head_weights.get(), state.entries,
        context->Input<Tensor>(2)->Data<int64_t>(),
        reinterpret_cast<const CudaT*>(context->Input<Tensor>(3)->Data<T>()),
        reinterpret_cast<const CudaT*>(context->Input<Tensor>(4)->Data<T>()),
        narrow<int>(batch), narrow<int>(sequence), narrow<int>(num_heads_), narrow<int>(head_size_),
        state.entry_count, narrow<int>(index_topk_), narrow<int>(compress_rate_), narrow<int>(rotary_dim_),
        narrow<int>(context->Input<Tensor>(3)->Shape()[1]), GetDeviceProp().maxThreadsPerBlock);
  }
 private:
  int64_t compress_rate_{};
  int64_t num_heads_{};
  int64_t head_size_{};
  int64_t index_topk_{};
  int64_t rotary_dim_{};
  float epsilon_{};
};

}  // namespace

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      LightningIndexer, kMSDomain, 1, T, kCudaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      LightningIndexer<T>);

REGISTER_KERNEL(MLFloat16)
REGISTER_KERNEL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
