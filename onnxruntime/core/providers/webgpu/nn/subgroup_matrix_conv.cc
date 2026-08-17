// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/nn/subgroup_matrix_conv.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/subgroup_matrix_common.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"
#include "core/providers/webgpu/nn/conv.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Subgroup-matrix implementation of the 1x1 / same-size Conv matmul. Runs its own
// Conv shape inference to decide eligibility (mirroring Conv::ComputeInternal), then
// builds the matmul operands and dispatches the shared subgroup-matrix kernel.
class SubgroupMatrixConvImpl final : public ConvOptImpl {
 public:
  SubgroupMatrixConvImpl(const ConvOptImplParent& parent, int32_t config_index,
                            SubgroupMatrixTilingSelector tiling_selector)
      : ConvOptImpl(parent), config_index_(config_index), tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, /*out*/ bool& handled) override {
    handled = false;

    const ConvAttributes& conv_attrs = parent_.ConvAttrs();
    const Activation& activation = parent_.ConvActivation();
    const bool is_channels_last = parent_.IsChannelsLast();
    const bool w_is_constant = parent_.IsWeightConstant();
    const Tensor* prepacked_kernel = parent_.PrepackedKernel();

    // Only a constant, channels-last weight is served here (constant so a runtime
    // transpose can be cached once; a prepacked weight is already transposed). It must
    // also be a plain, non-grouped matmul with no fused activation; anything else falls
    // back to the normal Conv path.
    if (!is_channels_last || !w_is_constant ||
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

    // Only pads/strides are needed to detect the 1x1 / same-size case.
    std::vector<uint32_t> strides, pads;
    auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
    std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
    std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);

    const auto input_height = input_shape[1];
    const auto input_width = input_shape[2];
    const auto input_channels = input_shape[3];
    const auto kernel_height = kernel_shape[2];
    const auto kernel_width = kernel_shape[3];

    const bool same_size = input_height == kernel_height && input_width == kernel_width && pads[0] == 0 && pads[1] == 0;
    const bool is_conv1x1_matmul = same_size || (kernel_height == 1 && kernel_width == 1 && pads[0] == 0 && pads[1] == 0 && strides[0] == 1 && strides[1] == 1);
    if (!is_conv1x1_matmul) {
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

std::unique_ptr<ConvOptImpl> CreateSubgroupMatrixConvImpl(const ConvOptImplParent& parent,
                                                             const ComputeContextBase& context) {
  int32_t config_index = 0;
  SubgroupMatrixTilingSelector tiling_selector;
  if (!TrySelectSubgroupMatrixConfig(context, config_index, tiling_selector)) {
    return nullptr;
  }
  return std::make_unique<SubgroupMatrixConvImpl>(parent, config_index, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
