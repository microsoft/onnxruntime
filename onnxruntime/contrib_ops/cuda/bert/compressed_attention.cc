// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/compressed_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
class CompressedAttention final : public CudaKernel {
 public:
  explicit CompressedAttention(const OpKernelInfo& info) : CudaKernel(info) {
    has_scale_ = info.GetAttr("scale", &scale_).IsOK();
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* query = context->Input<Tensor>(0);
    const Tensor* local = context->Input<Tensor>(1);
    const Tensor* compressed = context->Input<Tensor>(2);
    const Tensor* bias = context->Input<Tensor>(3);
    const Tensor* selected = context->Input<Tensor>(4);
    const Tensor* sink = context->Input<Tensor>(5);
    const Tensor* past_local = context->InputCount() > 6 ? context->Input<Tensor>(6) : nullptr;
    const Tensor* position_ids = context->InputCount() > 7 ? context->Input<Tensor>(7) : nullptr;
    ORT_RETURN_IF_NOT(query && local && sink && query->Shape().NumDimensions() == 4,
                      "query, local_kv, and head_sink are required.");
    const int64_t batch = query->Shape()[0];
    const int64_t heads = query->Shape()[1];
    const int64_t sequence = query->Shape()[2];
    const int64_t head_size = query->Shape()[3];
    ORT_RETURN_IF_NOT(local->Shape().NumDimensions() == 4 && local->Shape()[0] == batch &&
                          local->Shape()[1] == 1 && local->Shape()[3] == head_size,
                      "local_kv shape mismatch.");
    const bool fixed_cache = past_local != nullptr;
    ORT_RETURN_IF_NOT(fixed_cache == (position_ids != nullptr),
                      "past_local_kv and position_ids must be provided together.");
    const int64_t local_count = fixed_cache ? past_local->Shape()[2] : local->Shape()[2];
    if (fixed_cache) {
      ORT_RETURN_IF_NOT(local->Shape()[2] == sequence && past_local->Shape().NumDimensions() == 4 &&
                            past_local->Shape()[0] == batch && past_local->Shape()[1] == 1 &&
                            past_local->Shape()[3] == head_size && local_count > 0 &&
                            position_ids->Shape() == TensorShape({batch, sequence}),
                        "fixed local cache inputs have incompatible shapes.");
    }
    const int64_t compressed_count = compressed ? compressed->Shape()[2] : 0;
    if (compressed) ORT_RETURN_IF_NOT(compressed->Shape() == TensorShape({batch, 1, compressed_count, head_size}),
                                      "compressed_kv shape mismatch.");
    const int64_t selected_count = selected ? selected->Shape()[2] : compressed_count;
    if (selected) ORT_RETURN_IF_NOT(compressed && selected->Shape().NumDimensions() == 3 &&
                                        selected->Shape()[0] == batch && selected->Shape()[1] == sequence,
                                    "selected_indices shape mismatch.");
    const int64_t sink_count = sink->Shape().NumDimensions() == 0 ? 1 : sink->Shape()[0];
    ORT_RETURN_IF_NOT(sink->Shape().NumDimensions() <= 1 && (sink_count == 1 || sink_count == heads),
                      "head_sink shape mismatch.");
    int64_t bias_dims[4] = {1, 1, 1, 1};
    if (bias) {
      ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 4, "attention_bias must have rank 4.");
      const int64_t expected_keys = fixed_cache && bias->Shape()[3] == compressed_count
                    ? compressed_count
                    : local_count + compressed_count;
      const int64_t expected[4] = {batch, heads, sequence, expected_keys};
      for (int dim = 0; dim < 4; ++dim) {
        bias_dims[dim] = bias->Shape()[dim];
        ORT_RETURN_IF_NOT(bias_dims[dim] == 1 || bias_dims[dim] == expected[dim],
                          "attention_bias is not broadcastable.");
      }
    }
    using CudaT = typename ToCudaType<T>::MappedType;
    Tensor* output = context->Output(0, query->Shape());
    Tensor* present_local = fixed_cache ? context->Output(1, past_local->Shape()) : nullptr;
    if (fixed_cache && present_local->DataRaw() != past_local->DataRaw()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_local->MutableData<T>(), past_local->Data<T>(),
                         static_cast<size_t>(past_local->Shape().Size()) * sizeof(T),
                         cudaMemcpyDeviceToDevice, Stream(context)));
    }
    const float scale = has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
    return LaunchCompressedAttentionKernel<CudaT>(
      Stream(context), reinterpret_cast<CudaT*>(output->MutableData<T>()),
      fixed_cache ? reinterpret_cast<CudaT*>(present_local->MutableData<T>()) : nullptr,
        reinterpret_cast<const CudaT*>(query->Data<T>()), reinterpret_cast<const CudaT*>(local->Data<T>()),
      fixed_cache ? reinterpret_cast<const CudaT*>(past_local->Data<T>()) : nullptr,
      fixed_cache ? position_ids->Data<int64_t>() : nullptr,
        compressed ? reinterpret_cast<const CudaT*>(compressed->Data<T>()) : nullptr,
        bias ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr,
        selected ? selected->Data<int64_t>() : nullptr, reinterpret_cast<const CudaT*>(sink->Data<T>()),
        narrow<int>(batch), narrow<int>(heads), narrow<int>(sequence), narrow<int>(head_size),
        narrow<int>(local_count), narrow<int>(compressed_count), narrow<int>(selected_count),
        narrow<int>(sink_count), bias_dims[0], bias_dims[1], bias_dims[2], bias_dims[3], scale,
        fixed_cache, GetDeviceProp().maxThreadsPerBlock);
  }
 private:
  bool has_scale_{};
  float scale_{};
};

}  // namespace

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CompressedAttention, kMSDomain, 1, T, kCudaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()) \
          .MayInplace(6, 1),                                            \
      CompressedAttention<T>);

REGISTER_KERNEL(MLFloat16)
REGISTER_KERNEL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
