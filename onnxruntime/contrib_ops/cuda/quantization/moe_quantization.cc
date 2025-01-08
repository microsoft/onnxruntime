// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/moe_quantization.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL()                                                                  \
  ONNX_OPERATOR_KERNEL_EX(QMoE, kMSDomain, 1, kCudaExecutionProvider,                      \
                          (*KernelDefBuilder::Create())                                    \
                              .MayInplace(0, 0)                                            \
                              .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16>()) \
                              .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>()), \
                          QMoE);

REGISTER_KERNEL()

namespace {
template <typename T, bool use_quint4x2>
struct ToCudaTypeWrapper : public ToCudaType<T> {};

template <>
struct ToCudaTypeWrapper<uint8_t, false> {
  using MappedType = uint8_t;
};

template <>
struct ToCudaTypeWrapper<uint8_t, true> {
  using MappedType = cutlass::uint4b_t;
};

}  // anonymous namespace

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
}

template <typename CudaWeightT>
Status QMoE::QuantizedMoEImpl(OpKernelContext* context,
                              MoEParameters& moe_params,
                              const Tensor* input,
                              const Tensor* router_probs,
                              const Tensor* fc1_experts_weights,
                              const Tensor* fc1_experts_bias_optional,
                              const Tensor* fc2_experts_weights,
                              const Tensor* fc2_experts_bias_optional,
                              const Tensor* fc3_experts_weights_optional,
                              const Tensor* fc3_experts_bias_optional,
                              const Tensor* fc1_scales,
                              const Tensor* fc2_scales,
                              const Tensor* fc3_scales_optional,
                              const cudaDeviceProp& device_prop) const {
  auto stream = context->GetComputeStream();

  const int sm = device_prop.major * 10 + device_prop.minor;

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  using T = MLFloat16;
  using CudaT = typename ToCudaType<T>::MappedType;

  ort_fastertransformer::CutlassMoeFCRunner<CudaT, CudaWeightT> moe_runner(sm,
                                                                           fc3_experts_weights_optional != nullptr,
                                                                           normalize_routing_weights_,
                                                                           use_sparse_mixer_);

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
      static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.num_experts),
      static_cast<size_t>(k_));
  size_t fc2_output_size = k_ * moe_params.num_rows * moe_params.hidden_size * sizeof(CudaT);
  size_t expert_scales_size = k_ * moe_params.num_rows * sizeof(CudaT);
  size_t expanded_source_row_to_expanded_dest_row_size = k_ * moe_params.num_rows * sizeof(int);
  size_t expert_for_source_row_size = k_ * moe_params.num_rows * sizeof(int);

  IAllocatorUniquePtr<void> work_space = IAllocator::MakeUniquePtr<void>(allocator, ws_size, false, stream);
  IAllocatorUniquePtr<void> fc2_output = IAllocator::MakeUniquePtr<void>(allocator, fc2_output_size, false, stream);
  IAllocatorUniquePtr<void> expert_scales =
      IAllocator::MakeUniquePtr<void>(allocator, expert_scales_size, false, stream);
  IAllocatorUniquePtr<void> expanded_source_row_to_expanded_dest_row =
      IAllocator::MakeUniquePtr<void>(allocator, expanded_source_row_to_expanded_dest_row_size, false, stream);
  IAllocatorUniquePtr<void> expert_for_source_row =
      IAllocator::MakeUniquePtr<void>(allocator, expert_for_source_row_size, false, stream);

  moe_runner.run_moe_fc(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      reinterpret_cast<const CudaT*>(router_probs->template Data<T>()),
      reinterpret_cast<const CudaWeightT*>(fc1_experts_weights->DataRaw()),
      fc1_scales == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc1_scales->template Data<T>()),
      fc1_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
      activation_type_,
      fc3_experts_weights_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaWeightT*>(fc3_experts_weights_optional->DataRaw()),
      fc3_scales_optional == nullptr ? nullptr
                                     : reinterpret_cast<const CudaT*>(fc3_scales_optional->template Data<T>()),
      fc3_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc3_experts_bias_optional->template Data<T>()),
      reinterpret_cast<const CudaWeightT*>(fc2_experts_weights->DataRaw()),
      fc2_scales == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc2_scales->template Data<T>()),
      static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
      static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
      static_cast<int>(moe_params.local_num_experts), 0 /*local_experts_start_index_ used in sharded MoE*/,
      static_cast<int>(k_), reinterpret_cast<char*>(work_space.get()), reinterpret_cast<CudaT*>(fc2_output.get()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), Stream(context));

  Tensor* output = context->Output(0, input->Shape());

  ort_fastertransformer::finalize_moe_routing_kernelLauncher(
      reinterpret_cast<CudaT*>(fc2_output.get()), reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      fc2_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), static_cast<int>(moe_params.num_rows),
      static_cast<int>(moe_params.hidden_size), static_cast<int>(k_), Stream(context));

  return Status::OK();
}

Status QMoE::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_scales = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(5);
  const Tensor* fc2_scales = context->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(10);

  MoEQuantType quant_type = expert_weight_bits_ == 4 ? MoEQuantType::UINT4 : MoEQuantType::UINT8;
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(CheckInputs(moe_params, quant_type, input, router_probs, fc1_experts_weights,
                                  fc1_experts_bias_optional, fc2_experts_weights, fc2_experts_bias_optional,
                                  fc3_experts_weights_optional, fc3_experts_bias_optional));
  ORT_RETURN_IF_ERROR(CheckInputScales(fc1_scales, fc2_scales, fc3_scales_optional, moe_params.num_experts,
                                       moe_params.hidden_size, moe_params.inter_size));

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"  // Mute "maybe used uninitialized" warning for MoEParameters.
#endif

  if (quant_type == MoEQuantType::UINT4) {
    using CudaWeightT = typename ToCudaTypeWrapper<uint8_t, true>::MappedType;
    return QuantizedMoEImpl<CudaWeightT>(context, moe_params, input, router_probs,
                                         fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                         fc2_experts_bias_optional, fc3_experts_weights_optional,
                                         fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional,
                                         GetDeviceProp());
  } else {
    using CudaWeightT = typename ToCudaTypeWrapper<uint8_t, false>::MappedType;
    return QuantizedMoEImpl<CudaWeightT>(context, moe_params, input, router_probs,
                                         fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                         fc2_experts_bias_optional, fc3_experts_weights_optional,
                                         fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional,
                                         GetDeviceProp());
  }

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime