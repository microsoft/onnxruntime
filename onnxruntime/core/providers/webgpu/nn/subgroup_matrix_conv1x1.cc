// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/nn/subgroup_matrix_conv1x1.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Subgroup-matrix implementation of the 1x1 / same-size Conv matmul. Runs its own
// Conv shape inference to decide eligibility (mirroring Conv::ComputeInternal), then
// builds the matmul operands and dispatches the shared subgroup-matrix kernel.
class SubgroupMatrixConv1x1Impl final : public Conv1x1OptImpl {
 public:
  SubgroupMatrixConv1x1Impl(int32_t config_index, SubgroupMatrixTilingSelector tiling_selector)
      : config_index_(config_index), tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, const ConvAttributes& conv_attrs,
                 const Activation& activation, bool is_channels_last,
                 const Tensor* prepacked_kernel, bool w_is_constant,
                 /*out*/ bool& handled) override {
    handled = false;

    // The subgroup-matrix kernel computes a plain, non-grouped matmul; decline
    // anything it can't serve so the caller falls back to the normal Conv path.
    if (activation.activation_kind_ != ActivationKind::None || conv_attrs.group != 1) {
      return Status::OK();
    }

    // Read the Conv operands from context. prepacked_kernel (when non-null) is the
    // prepacked OHWI->HWIO weight; otherwise the original weight is input 1.
    const auto* input = context.Input<Tensor>(0);
    const bool kernel_is_prepacked = prepacked_kernel != nullptr;
    const Tensor* kernel = kernel_is_prepacked ? prepacked_kernel : context.Input<Tensor>(1);
    const Tensor* bias = context.InputCount() > 2 ? context.Input<Tensor>(2) : nullptr;

    // Shape inference mirrors Conv::ComputeInternal so eligibility and the operand
    // layout are computed identically.
    TensorShape input_shape = input->Shape();
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
    TensorShape input_spacial_shape = is_channels_last ? TensorShape(TensorShapeVector(std::next(input_shape_vector.begin()), std::prev(input_shape_vector.end()))) : input_shape.Slice(2);
    ORT_RETURN_IF_ERROR(conv_attrs.InferPadsAndOutputShape(input_spacial_shape, kernel_spacial_shape_vector, local_strides, local_dilations, local_pads, output_shape_vector));
    const auto output_channels = kernel_shape[0];
    if (is_channels_last) {
      output_shape_vector.push_back(output_channels);
    } else {
      output_shape_vector.insert(output_shape_vector.begin() + 1, output_channels);
    }
    const TensorShape output_shape = TensorShape(output_shape_vector);

    // Only pads/strides are needed to detect the 1x1 / same-size case.
    std::vector<uint32_t> strides, pads;
    auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
    std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
    std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);

    const auto rank = input_shape.NumDimensions();
    if (rank == 3) {
      // Conv1D: reshape to a height-1 Conv2D (mirrors Conv::ComputeInternal).
      TensorShapeVector kernel_shape_vector = kernel_shape.AsShapeVector();
      input_shape_vector.insert(input_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
      output_shape_vector.insert(output_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
      kernel_shape_vector.insert(kernel_shape_vector.begin() + 2, 1);
      input_shape = TensorShape(input_shape_vector);
      kernel_shape = TensorShape(kernel_shape_vector);
      pads.insert(pads.begin(), 0);
      pads.insert(pads.begin() + 2, 0);
      strides.insert(strides.begin(), 1);
    } else if (rank != 4) {
      // Conv3D / unsupported rank: not a 1x1 matmul candidate.
      return Status::OK();
    }

    const auto input_height = input_shape[is_channels_last ? 1 : 2];
    const auto input_width = input_shape[is_channels_last ? 2 : 3];
    const auto input_channels = input_shape[is_channels_last ? 3 : 1];
    const auto kernel_height = kernel_shape[2];
    const auto kernel_width = kernel_shape[3];

    const bool same_size = is_channels_last && input_height == kernel_height && input_width == kernel_width && pads[0] == 0 && pads[1] == 0;
    const bool is_conv1x1_matmul = same_size || (kernel_height == 1 && kernel_width == 1 && pads[0] == 0 && pads[1] == 0 && strides[0] == 1 && strides[1] == 1);
    if (!is_conv1x1_matmul) {
      return Status::OK();
    }

    auto* output = context.Output(0, output_shape);

    // Build the matmul operands. Channels-last transposes the non-prepacked weight and
    // uses the activation as A; channels-first uses the weight as A and activation as B.
    const InlinedVector<size_t> perm = {2, 3, 1, 0};
    Tensor transposed_kernel;
    const Tensor* a = nullptr;
    const Tensor* b = nullptr;
    TensorShape a_shape;
    TensorShape b_shape;
    if (is_channels_last) {
      const Tensor* matmul_kernel = kernel;
      if (!kernel_is_prepacked) {
        ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
        matmul_kernel = &transposed_kernel;
      }
      a = input;
      b = matmul_kernel;
      if (same_size) {
        const auto shared_dim = input_height * input_width * input_channels;
        a_shape = TensorShape({1, batch, shared_dim});
        b_shape = TensorShape({1, shared_dim, output_channels});
      } else {
        a_shape = TensorShape({batch, input_height * input_width, input_channels});
        b_shape = TensorShape({1, input_channels, output_channels});
      }
    } else {
      a = kernel;
      b = input;
      a_shape = TensorShape({1, output_channels, input_channels});
      b_shape = TensorShape({batch, input_channels, input_height * input_width});
    }

    // Fold A's leading dims into M when B is a shared 2D weight so the reshaped
    // operands collapse into a single 2D-weight matmul the shared dispatch handles.
    if (a_shape.NumDimensions() >= 2 && b_shape.NumDimensions() >= 2) {
      const int64_t batch_a = a_shape.SizeToDimension(a_shape.NumDimensions() - 2);
      const int64_t batch_b = b_shape.SizeToDimension(b_shape.NumDimensions() - 2);
      if (batch_a != 1 && batch_b == 1) {
        const int64_t k = a_shape[a_shape.NumDimensions() - 1];
        const int64_t n = b_shape[b_shape.NumDimensions() - 1];
        const int64_t batch_and_m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
        a_shape = TensorShape({batch_and_m, k});
        b_shape = TensorShape({k, n});
      }
    }

    return DispatchSubgroupMatrixMatMul(context, config_index_, tiling_selector_, pad_cache_,
                                        a, b, bias, output, a_shape, b_shape,
                                        is_channels_last && w_is_constant, handled);
  }

 private:
  const int32_t config_index_;
  const SubgroupMatrixTilingSelector tiling_selector_;

  // Odd-N even-strided B cache, shared with the subgroup-matrix MatMul path.
  mutable SubgroupMatrixPadBCache pad_cache_;
};

}  // namespace

std::unique_ptr<Conv1x1OptImpl> CreateSubgroupMatrixConv1x1Impl(const ComputeContextBase& context) {
  // Only run on devices that report the fixed 8x16x16 F16 subgroup-matrix config
  // this kernel is implemented for.
  int32_t config_index = 0;
  if (!IsSubgroupMatrixConfigSupported(context, /*is_fp16=*/true, config_index) ||
      !supported_subgroup_matrix_configs[config_index].Is(8, 16, 16)) {
    return nullptr;
  }
  // Intel GPUs use a tuned/heuristic tiling policy; every other vendor falls back
  // to a fixed default tiling.
  const bool is_intel = context.AdapterInfo().vendor == std::string_view{"intel"};
  SubgroupMatrixTilingSelector tiling_selector =
      is_intel ? intel::CreateSubgroupMatrixTilingSelector(context) : MakeDefaultSubgroupMatrixTilingSelector();
  if (!tiling_selector) {
    return nullptr;
  }
  return std::make_unique<SubgroupMatrixConv1x1Impl>(config_index, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
