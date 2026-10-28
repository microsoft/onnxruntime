// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_low_rank_mix.h"

#include <limits>
#include <type_traits>

#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "contrib_ops/cuda/bert/hyper_connection_low_rank_mix_impl.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

#define REGISTER_HYPER_CONNECTION_LOW_RANK_MIX_CUDA_KERNEL(T)             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                          \
      HyperConnectionLowRankMix, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create())                                       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),         \
      HyperConnectionLowRankMix<T>);

REGISTER_HYPER_CONNECTION_LOW_RANK_MIX_CUDA_KERNEL(float)
REGISTER_HYPER_CONNECTION_LOW_RANK_MIX_CUDA_KERNEL(MLFloat16)
REGISTER_HYPER_CONNECTION_LOW_RANK_MIX_CUDA_KERNEL(BFloat16)

#undef REGISTER_HYPER_CONNECTION_LOW_RANK_MIX_CUDA_KERNEL

template <typename T>
HyperConnectionLowRankMix<T>::HyperConnectionLowRankMix(const OpKernelInfo& info)
    : CudaKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      reduction_scale_(info.GetAttrOrDefault<float>("reduction_scale", 1.0f)),
      silu_alpha_(info.GetAttrOrDefault<float>("silu_alpha", 1.0f)),
      gate_div_(info.GetAttrOrDefault<float>("gate_div", 1.0f)),
      gate_mul_(info.GetAttrOrDefault<float>("gate_mul", 1.0f)) {
  ORT_ENFORCE(num_branches_ > 0, "HyperConnectionLowRankMix requires a positive num_branches");
  ORT_ENFORCE(gate_div_ != 0.0f, "gate_div must be non-zero");
  const Tensor* constant_input = nullptr;
  weights_are_constant_ = info.TryGetConstantInput(1, &constant_input) &&
                          info.TryGetConstantInput(2, &constant_input) &&
                          info.TryGetConstantInput(3, &constant_input) &&
                          info.TryGetConstantInput(4, &constant_input);
}

template <typename T>
Status HyperConnectionLowRankMix<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;

  if constexpr (std::is_same_v<T, BFloat16>) {
    ORT_RETURN_IF_NOT(GetDeviceProp().major >= 8,
                      "HyperConnectionLowRankMix with BFloat16 requires CUDA compute capability 8.0 or higher");
  }

  const Tensor* streams = context->Input<Tensor>(0);
  const Tensor* norm_scale = context->Input<Tensor>(1);
  const Tensor* down_weight = context->Input<Tensor>(2);
  const Tensor* up_weight = context->Input<Tensor>(3);
  const Tensor* inject_weight = context->Input<Tensor>(4);
  const Tensor* prev_block_output = context->Input<Tensor>(5);
  const Tensor* prev_post_gate = context->Input<Tensor>(6);

  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_NOT(params.branches <= std::numeric_limits<int>::max() &&
                        params.hidden <= std::numeric_limits<int>::max() / params.branches,
                    "branch or hidden dimension is too large for CUDA");
  const int64_t branches = params.branches;
  const int64_t hidden = params.hidden;
  const int64_t features = branches * hidden;

  ORT_RETURN_IF_NOT(norm_scale->Shape().Size() == features && norm_scale->Shape().NumDimensions() == 1,
                    "norm_scale must have shape (num_branches * hidden)");
  ORT_RETURN_IF_NOT(down_weight->Shape().NumDimensions() == 2 && down_weight->Shape()[0] == features,
                    "down_weight must have shape (num_branches * hidden, rank)");
  const int64_t rank = down_weight->Shape()[1];
  ORT_RETURN_IF_NOT(rank > 0 && rank <= std::numeric_limits<int>::max(), "rank must be positive");
  ORT_RETURN_IF_NOT(up_weight->Shape().NumDimensions() == 2 && up_weight->Shape()[0] == rank &&
                        up_weight->Shape()[1] == features,
                    "up_weight must have shape (rank, num_branches * hidden)");
  ORT_RETURN_IF_NOT(inject_weight->Shape().NumDimensions() == 2 && inject_weight->Shape()[0] == features &&
                        inject_weight->Shape()[1] == branches,
                    "inject_weight must have shape (num_branches * hidden, num_branches)");
  ORT_RETURN_IF_NOT((prev_block_output == nullptr) == (prev_post_gate == nullptr),
                    "prev_block_output and prev_post_gate must be provided together");

  const bool has_prologue = prev_block_output != nullptr;
  if (has_prologue) {
    const auto& dims = streams->Shape().GetDims();
    TensorShapeVector expected_output(dims.begin(), dims.begin() + params.prefix_rank);
    expected_output.push_back(hidden);
    TensorShapeVector expected_gate(dims.begin(), dims.begin() + params.prefix_rank);
    expected_gate.push_back(branches);
    ORT_RETURN_IF_NOT(prev_block_output->Shape() == TensorShape(expected_output),
                      "prev_block_output must have shape (..., hidden)");
    ORT_RETURN_IF_NOT(prev_post_gate->Shape() == TensorShape(expected_gate),
                      "prev_post_gate must have shape (..., num_branches)");
  }

  Tensor* pre_output = context->Output(0, TensorShape(params.reduced_shape));
  TensorShapeVector gate_shape(params.reduced_shape.begin(), params.reduced_shape.end() - 1);
  gate_shape.push_back(branches);
  Tensor* post_gate = context->Output(1, TensorShape(gate_shape));
  Tensor* streams_out = context->Output(2, streams->Shape());

  if (params.rows == 0) {
    return Status::OK();
  }

  HyperConnectionLowRankMixConfig config;
  config.rows = params.rows;
  config.branches = static_cast<int>(branches);
  config.hidden = static_cast<int>(hidden);
  config.rank = static_cast<int>(rank);
  config.epsilon = epsilon_;
  config.reduction_scale = reduction_scale_;
  config.silu_alpha = silu_alpha_;
  config.gate_div = gate_div_;
  config.gate_mul = gate_mul_;

  IAllocatorUniquePtr<T> streams_scratch;
  HyperConnectionLowRankMixTensors<CudaT> tensors;
  tensors.streams = reinterpret_cast<const CudaT*>(streams->Data<T>());
  tensors.norm_scale = reinterpret_cast<const CudaT*>(norm_scale->Data<T>());
  tensors.down_weight = reinterpret_cast<const CudaT*>(down_weight->Data<T>());
  tensors.up_weight = reinterpret_cast<const CudaT*>(up_weight->Data<T>());
  tensors.inject_weight = reinterpret_cast<const CudaT*>(inject_weight->Data<T>());
  if (has_prologue) {
    tensors.prev_block_output = reinterpret_cast<const CudaT*>(prev_block_output->Data<T>());
    tensors.prev_post_gate = reinterpret_cast<const CudaT*>(prev_post_gate->Data<T>());
    if (streams_out != nullptr) {
      tensors.streams_out = reinterpret_cast<CudaT*>(streams_out->MutableData<T>());
    }
  } else if (streams_out != nullptr) {
    // Without a prologue the streams pass through unchanged.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(streams_out->MutableDataRaw(), streams->DataRaw(),
                                         streams->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(context)));
  }
  tensors.pre_output = reinterpret_cast<CudaT*>(pre_output->MutableData<T>());
  tensors.post_gate = reinterpret_cast<CudaT*>(post_gate->MutableData<T>());

  // The fused kernels use 16-byte vector accesses on every tensor.
  auto aligned16 = [](const void* ptr) { return ptr == nullptr || reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
  const bool aligned = aligned16(tensors.streams) && aligned16(tensors.norm_scale) &&
                       aligned16(tensors.down_weight) && aligned16(tensors.up_weight) &&
                       aligned16(tensors.inject_weight) && aligned16(tensors.prev_block_output) &&
                       aligned16(tensors.streams_out);

  const auto& device_prop = GetDeviceProp();
  if (aligned && CanUseFusedHyperConnectionLowRankMix<CudaT>(device_prop, config)) {
    if (has_prologue && tensors.streams_out == nullptr) {
      // HcUpPreKernel reads the post-mixed streams written by HcDownKernel; allocate scratch only for the fused path.
      streams_scratch = GetScratchBuffer<T>(static_cast<size_t>(params.rows * features), GetComputeStream(context));
      tensors.streams_out = reinterpret_cast<CudaT*>(streams_scratch.get());
    }
    auto scratch = GetScratchBuffer<float>(HyperConnectionLowRankMixFusedScratchElements(config),
                                           GetComputeStream(context));
    return LaunchHyperConnectionLowRankMixFused<CudaT>(device_prop, Stream(context), config, tensors,
                                                       scratch.get(), weights_are_constant_);
  }

  const size_t rows = static_cast<size_t>(params.rows);
  auto hn = GetScratchBuffer<T>(rows * static_cast<size_t>(features), GetComputeStream(context));
  auto down = GetScratchBuffer<T>(rows * static_cast<size_t>(rank), GetComputeStream(context));
  auto up = GetScratchBuffer<T>(rows * static_cast<size_t>(features), GetComputeStream(context));
  return LaunchHyperConnectionLowRankMixUnfused<CudaT>(
      GetCublasHandle(context), Stream(context), config, tensors, reinterpret_cast<CudaT*>(hn.get()),
      reinterpret_cast<CudaT*>(down.get()), reinterpret_cast<CudaT*>(up.get()));
}

template class HyperConnectionLowRankMix<float>;
template class HyperConnectionLowRankMix<MLFloat16>;
template class HyperConnectionLowRankMix<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
