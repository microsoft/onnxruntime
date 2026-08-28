// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/nn/subgroup_matrix_conv.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/subgroup_matrix_common.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Subgroup-matrix implementation of the 1x1 / same-size Conv matmul. Runs its own
// Conv shape inference to decide eligibility (mirroring Conv::ComputeInternal), then
// builds the matmul operands and dispatches the shared subgroup-matrix kernel.
template <bool is_channels_last, bool is_fused>
class SubgroupMatrixConvImpl final : public Conv<is_channels_last, is_fused>::ConvOptImpl {
  using ConvT = Conv<is_channels_last, is_fused>;
  using Base = typename ConvT::ConvOptImpl;

 public:
  SubgroupMatrixConvImpl(const ConvT& parent, int32_t config_index,
                         SubgroupMatrixTilingSelector tiling_selector)
      : Base(parent), config_index_(config_index), tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, /*out*/ bool& handled) override {
    handled = false;

    const ConvT& parent = this->parent_;
    const ConvAttributes& conv_attrs = parent.ConvAttrs();
    const Activation& activation = parent.ConvActivation();
    const bool w_is_constant = parent.IsWeightConstant();
    const Tensor* prepacked_kernel = parent.PrepackedKernel();

    // Only a constant weight is served here (so a runtime transpose can be cached once;
    // a prepacked weight is already transposed). It must also be a plain, non-grouped
    // matmul with no fused activation; anything else falls back to the normal Conv path.
    // (channels-last is guaranteed by CreateSubgroupMatrixConvImpl.)
    if (!w_is_constant ||
        activation.activation_kind_ != ActivationKind::None || conv_attrs.group != 1) {
      return Status::OK();
    }

    const auto* input = context.Input<Tensor>(0);
    // A prepacked weight is the OIHW->HWIO transposed kernel; otherwise read the
    // original OIHW weight from input 1.
    const bool kernel_is_prepacked = prepacked_kernel != nullptr;
    const Tensor* kernel = kernel_is_prepacked ? prepacked_kernel : context.Input<Tensor>(1);
    const Tensor* bias = context.InputCount() > 2 ? context.Input<Tensor>(2) : nullptr;

    // Shape inference mirrors Conv::ComputeInternal (channels-last). Only Conv2D (a
    // rank-4 NHWC input) is supported.
    TensorShape input_shape = input->Shape();
    if (input_shape.NumDimensions() != 4) {
      return Status::OK();
    }
    TensorShape kernel_shape = kernel_is_prepacked
                                   ? TensorShape(TensorShapeVector{kernel->Shape()[3], kernel->Shape()[2], kernel->Shape()[0], kernel->Shape()[1]})
                                   : kernel->Shape();
    ConvAttributes::ConvPadVector local_pads(conv_attrs.pads.begin(), conv_attrs.pads.end());
    TensorShapeVector local_dilations(conv_attrs.dilations.begin(), conv_attrs.dilations.end());
    TensorShapeVector local_strides(conv_attrs.strides.begin(), conv_attrs.strides.end());
    TensorShapeVector kernel_spacial_shape_vector;
    ORT_RETURN_IF_ERROR(conv_attrs.ComputeKernelShape(kernel_shape, kernel_spacial_shape_vector, false));
    if (local_pads.empty()) {
      local_pads.resize(kernel_spacial_shape_vector.size() * 2, 0);
    }
    if (local_dilations.empty()) {
      local_dilations.resize(kernel_spacial_shape_vector.size(), 1);
    }
    if (local_strides.empty()) {
      local_strides.resize(kernel_spacial_shape_vector.size(), 1);
    }
    TensorShapeVector input_shape_vector = input_shape.AsShapeVector();
    const auto batch = input_shape[0];
    TensorShapeVector output_shape_vector = {batch};
    TensorShape input_spacial_shape = TensorShape(TensorShapeVector(std::next(input_shape_vector.begin()), std::prev(input_shape_vector.end())));
    ORT_RETURN_IF_ERROR(conv_attrs.InferPadsAndOutputShape(input_spacial_shape, kernel_spacial_shape_vector, local_strides, local_dilations, local_pads, output_shape_vector));
    const auto output_channels = kernel_shape[0];
    output_shape_vector.push_back(output_channels);
    const TensorShape output_shape = TensorShape(output_shape_vector);

    const auto input_height = input_shape[1];
    const auto input_width = input_shape[2];
    const auto input_channels = input_shape[3];
    const auto kernel_height = kernel_shape[2];
    const auto kernel_width = kernel_shape[3];
    const auto output_height = output_shape_vector[1];
    const auto output_width = output_shape_vector[2];

    // Both reshapes need all pads (not just the leading ones) and the inferred output
    // geometry to qualify; see the predicates in conv.h, which the normal Conv MatMul
    // path shares.
    const bool same_size = IsConvSameSizeMatMul(input_height, input_width, kernel_height, kernel_width,
                                                output_height, output_width, local_pads);
    const bool is_conv_matmul =
        same_size || IsConv1x1MatMul(kernel_height, kernel_width, local_pads, local_strides);
    if (!is_conv_matmul) {
      return Status::OK();
    }

    auto* output = context.Output(0, output_shape);

    // B is the OIHW->HWIO transposed weight. A prepacked weight is already transposed;
    // otherwise transpose the constant weight once and reuse the cached copy.
    const Tensor* weight = prepacked_kernel;
    if (weight == nullptr) {
      ORT_RETURN_IF_ERROR(EnsureTransposedKernel(context, kernel, kernel_shape));
      weight = cached_transposed_kernel_.get();
    }

    // The operands are a plain 2D weight matmul (M x K) @ (K x N): a 1x1 conv makes each
    // N,H,W position an independent row; same-size folds the whole window into one row
    // per batch element.
    TensorShape a_shape;
    TensorShape b_shape;
    if (same_size) {
      const auto shared_dim = input_height * input_width * input_channels;
      a_shape = TensorShape({batch, shared_dim});
      b_shape = TensorShape({shared_dim, output_channels});
    } else {
      a_shape = TensorShape({batch * input_height * input_width, input_channels});
      b_shape = TensorShape({input_channels, output_channels});
    }

    return DispatchSubgroupMatrixMatMul(context, config_index_, tiling_selector_, pad_cache_,
                                        input, weight, bias, output, a_shape, b_shape,
                                        /*b_is_constant=*/true, handled);
  }

 private:
  // Builds the OIHW->HWIO transposed weight on first use and caches it; only valid
  // because the weight is a constant initializer (checked by the caller).
  Status EnsureTransposedKernel(ComputeContext& context, const Tensor* kernel,
                                const TensorShape& kernel_shape) const {
    std::call_once(transpose_once_, [&]() {
      const InlinedVector<size_t> perm = {2, 3, 1, 0};
      auto transposed = std::make_unique<Tensor>();
      if (TransposeKernel(context, kernel, kernel_shape, transposed.get(), perm).IsOK()) {
        cached_transposed_kernel_ = std::move(transposed);
      }
    });
    return cached_transposed_kernel_ ? Status::OK()
                                     : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to transpose Conv 1x1 weight.");
  }

  const int32_t config_index_;
  const SubgroupMatrixTilingSelector tiling_selector_;

  // OIHW->HWIO transposed constant weight, built once by EnsureTransposedKernel.
  mutable std::once_flag transpose_once_;
  mutable std::unique_ptr<Tensor> cached_transposed_kernel_;

  // Odd-N even-strided B cache, shared with the subgroup-matrix MatMul path.
  mutable SubgroupMatrixPadBCache pad_cache_;
};

}  // namespace

template <bool is_channels_last, bool is_fused>
std::unique_ptr<typename Conv<is_channels_last, is_fused>::ConvOptImpl> CreateSubgroupMatrixConvImpl(
    const Conv<is_channels_last, is_fused>& parent,
    const ComputeContextBase& context) {
  // Only channels-last Conv is served by the subgroup-matrix path; the caller falls back
  // to the normal Conv path for channels-first.
  if constexpr (!is_channels_last) {
    ORT_UNUSED_PARAMETER(parent);
    ORT_UNUSED_PARAMETER(context);
    return nullptr;
  } else {
    int32_t config_index = 0;
    SubgroupMatrixTilingSelector tiling_selector;
    if (!TrySelectSubgroupMatrixConfig(context, config_index, tiling_selector)) {
      return nullptr;
    }
    return std::make_unique<SubgroupMatrixConvImpl<is_channels_last, is_fused>>(parent, config_index,
                                                                                std::move(tiling_selector));
  }
}

// Explicit instantiation for every Conv instantiation (see conv.cc).
#define WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(CHANNELS_LAST, FUSED)              \
  template std::unique_ptr<Conv<CHANNELS_LAST, FUSED>::ConvOptImpl>                            \
  CreateSubgroupMatrixConvImpl<CHANNELS_LAST, FUSED>(const Conv<CHANNELS_LAST, FUSED>& parent, \
                                                     const ComputeContextBase& context);

WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(false, false)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(false, true)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(true, false)
WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL(true, true)

#undef WEBGPU_INSTANTIATE_CREATE_SUBGROUP_MATRIX_CONV_IMPL

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
