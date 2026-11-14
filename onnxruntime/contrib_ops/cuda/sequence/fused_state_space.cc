// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/sequence/fused_state_space_impl.h"

#include <cmath>
#include <limits>

namespace onnxruntime::contrib::cuda {

using onnxruntime::cuda::CudaKernel;

class Mamba2Verify final : public CudaKernel {
 public:
  explicit Mamba2Verify(const OpKernelInfo& info) : CudaKernel(info) {
    heads_ = narrow<int>(info.GetAttrOrDefault<int64_t>("heads", 0));
    groups_ = narrow<int>(info.GetAttrOrDefault<int64_t>("n_groups", 0));
    ORT_ENFORCE(heads_ > 0 && groups_ > 0 && heads_ % groups_ == 0 &&
                    heads_ <= std::numeric_limits<int>::max() / 1024,
                "Mamba2Verify requires heads > 0, n_groups > 0, heads % n_groups == 0 "
                "and heads <= INT_MAX / 1024, got heads=",
                heads_, ", n_groups=", groups_);
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("head_dim", 64) == 64 &&
                    info.GetAttrOrDefault<int64_t>("d_state", 128) == 128,
                "Mamba2Verify currently requires head_dim=64, d_state=128");
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
    time_step_min_ = info.GetAttrOrDefault<float>("time_step_min", 0.001f);
    ORT_ENFORCE(std::isfinite(epsilon_) && epsilon_ > 0 &&
                std::isfinite(time_step_min_) && time_step_min_ >= 0);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Mamba2Verify);

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* projection = context->Input<Tensor>(0);
    const auto* conv = context->Input<Tensor>(1);
    const auto* ssm = context->Input<Tensor>(2);
    const auto* state_row = context->Input<Tensor>(9);
    const auto* cu_seqlens = context->InputCount() > 10 ? context->Input<Tensor>(10) : nullptr;
    const auto* state_slots = context->InputCount() > 11 ? context->Input<Tensor>(11) : nullptr;
    const int di = heads_ * 64;
    const int channels = di + 2 * groups_ * 128;
    ORT_RETURN_IF_NOT(projection->Shape().NumDimensions() == 2, "projection must have rank 2");
    const int64_t rows64 = projection->Shape()[0];
    ORT_RETURN_IF_NOT(rows64 >= 1 && rows64 <= std::numeric_limits<int>::max() &&
                          projection->Shape()[1] == di + channels + heads_,
                      "Invalid projection shape");
    const int rows = static_cast<int>(rows64);
    const bool banked = cu_seqlens != nullptr || state_slots != nullptr || conv->Shape().NumDimensions() == 4;
    ORT_RETURN_IF_NOT(!banked || (cu_seqlens != nullptr && state_slots != nullptr),
                      "Multi-sequence Mamba2Verify requires both cumulative_sequence_lengths and state_slots");

    int64_t num_banks = 1;
    int64_t slots = 0;
    int batch_size = 1;
    if (!banked) {
      ORT_RETURN_IF_NOT(conv->Shape().NumDimensions() == 3 && conv->Shape()[0] >= 1 &&
                            conv->Shape()[1] == channels && conv->Shape()[2] == 3,
                        "Invalid convolution state shape");
      slots = conv->Shape()[0];
      ORT_RETURN_IF_NOT(ssm->Shape() == TensorShape({slots, heads_, 128, 64}), "Invalid SSM state shape");
    } else {
      ORT_RETURN_IF_NOT(conv->Shape().NumDimensions() == 4 && conv->Shape()[0] >= 1 && conv->Shape()[1] >= 1 &&
                            conv->Shape()[2] == channels && conv->Shape()[3] == 3,
                        "Invalid banked convolution state shape");
      num_banks = conv->Shape()[0];
      slots = conv->Shape()[1];
      ORT_RETURN_IF_NOT(ssm->Shape() == TensorShape({num_banks, slots, heads_, 128, 64}),
                        "Invalid banked SSM state shape");
      ORT_RETURN_IF_NOT(state_row->Shape().NumDimensions() == 1 && state_row->Shape()[0] >= 1 &&
                            state_row->Shape()[0] < std::numeric_limits<int>::max(),
                        "Invalid batched state_row shape");
      batch_size = static_cast<int>(state_row->Shape()[0]);
      ORT_RETURN_IF_NOT(cu_seqlens->Shape() == TensorShape({static_cast<int64_t>(batch_size) + 1}) &&
                            state_slots->Shape() == TensorShape({static_cast<int64_t>(batch_size)}),
                        "Invalid cumulative_sequence_lengths or state_slots shape");
    }

    const auto& conv_weight_shape = context->Input<Tensor>(3)->Shape();
    ORT_RETURN_IF_NOT((conv_weight_shape == TensorShape({channels, 4}) ||
                       conv_weight_shape == TensorShape({channels, 1, 4})) &&
                          context->Input<Tensor>(4)->Shape() == TensorShape({channels}),
                      "Invalid convolution parameters");
    for (int i = 5; i <= 7; ++i) {
      ORT_RETURN_IF_NOT(context->Input<Tensor>(i)->Shape() == TensorShape({heads_}), "Invalid head parameter");
    }
    ORT_RETURN_IF_NOT(context->Input<Tensor>(8)->Shape() == TensorShape({di}) &&
                          (banked || state_row->Shape() == TensorShape({1})),
                      "Invalid norm weight or state_row");
    auto* y = context->Output(0, {rows, di});
    auto* conv_out = context->Output(1, conv->Shape());
    auto* ssm_out = context->Output(2, ssm->Shape());
    auto* xbc = context->OutputCount() > 3 ? context->Output(3, {rows, channels}) : nullptr;
    auto* readout = context->OutputCount() > 4 ? context->Output(4, {rows, di}) : nullptr;
    // Both intermediates remain kernel workspace even when their public outputs are omitted.
    const size_t xbc_count = SafeInt<size_t>(rows) * channels;
    const size_t readout_count = SafeInt<size_t>(rows) * di;
    auto xbc_scratch = GetScratchBuffer<BFloat16>(
        xbc ? 0 : xbc_count, GetComputeStream(context));
    auto readout_scratch = GetScratchBuffer<float>(
        readout ? 0 : readout_count, GetComputeStream(context));
    auto* xbc_data = xbc ? xbc->MutableData<BFloat16>() : xbc_scratch.get();
    auto* readout_data = readout ? readout->MutableData<float>() : readout_scratch.get();
    if (!banked) {
      // Unwritten snapshot slots retain their input values, including when ORT does not alias buffers.
      if (slots > rows && conv_out->MutableDataRaw() != conv->DataRaw()) {
        const size_t offset = conv->SizeInBytes() / static_cast<size_t>(slots) * rows;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(static_cast<uint8_t*>(conv_out->MutableDataRaw()) + offset,
                                             static_cast<const uint8_t*>(conv->DataRaw()) + offset,
                                             conv->SizeInBytes() - offset, cudaMemcpyDeviceToDevice, Stream(context)));
      }
      if (slots > rows && ssm_out->MutableDataRaw() != ssm->DataRaw()) {
        const size_t offset = ssm->SizeInBytes() / static_cast<size_t>(slots) * rows;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(static_cast<uint8_t*>(ssm_out->MutableDataRaw()) + offset,
                                             static_cast<const uint8_t*>(ssm->DataRaw()) + offset,
                                             ssm->SizeInBytes() - offset, cudaMemcpyDeviceToDevice, Stream(context)));
      }
      CUDA_RETURN_IF_ERROR(LaunchMamba2Verify(
          Stream(context), rows, slots, heads_, groups_, epsilon_, time_step_min_, projection->DataRaw(),
          conv->DataRaw(), ssm->Data<float>(), context->Input<Tensor>(3)->DataRaw(),
          context->Input<Tensor>(4)->DataRaw(), context->Input<Tensor>(5)->Data<float>(),
          context->Input<Tensor>(6)->Data<float>(), context->Input<Tensor>(7)->Data<float>(),
          context->Input<Tensor>(8)->Data<float>(), state_row->Data<int32_t>(), y->MutableDataRaw(),
          conv_out->MutableDataRaw(), ssm_out->MutableData<float>(), xbc_data, readout_data));
    } else {
      // Unreferenced banks and unwritten trailing snapshot rows retain their input values when not aliased.
      if (conv_out->MutableDataRaw() != conv->DataRaw()) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(conv_out->MutableDataRaw(), conv->DataRaw(), conv->SizeInBytes(),
                                             cudaMemcpyDeviceToDevice, Stream(context)));
      }
      if (ssm_out->MutableDataRaw() != ssm->DataRaw()) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(ssm_out->MutableDataRaw(), ssm->DataRaw(), ssm->SizeInBytes(),
                                             cudaMemcpyDeviceToDevice, Stream(context)));
      }
      CUDA_RETURN_IF_ERROR(LaunchMamba2VerifyBatched(
          Stream(context), rows, batch_size, num_banks, slots, heads_, groups_, epsilon_, time_step_min_,
          projection->DataRaw(), conv->DataRaw(), ssm->Data<float>(), context->Input<Tensor>(3)->DataRaw(),
          context->Input<Tensor>(4)->DataRaw(), context->Input<Tensor>(5)->Data<float>(),
          context->Input<Tensor>(6)->Data<float>(), context->Input<Tensor>(7)->Data<float>(),
          context->Input<Tensor>(8)->Data<float>(), state_row->Data<int32_t>(), cu_seqlens->Data<int32_t>(),
          state_slots->Data<int32_t>(), y->MutableDataRaw(), conv_out->MutableDataRaw(),
          ssm_out->MutableData<float>(), xbc_data, readout_data));
    }
    return Status::OK();
  }

 private:
  int heads_, groups_;
  float epsilon_, time_step_min_;
};

ONNX_OPERATOR_KERNEL_EX(
    Mamba2Verify, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int32_t>())
        .MayInplace(1, 1)
        .MayInplace(2, 2),
    Mamba2Verify);
}  // namespace onnxruntime::contrib::cuda
